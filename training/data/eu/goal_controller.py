"""One bounded scheduler tick; durable claims prevent duplicate campaign dispatch.

The caller supplies freshly loaded source snapshots on *every* tick. Collection
workers exclusively own source repositories; this module writes only a separate
HFStore control repository, never corpus rows. A completed call's reservation is
kept for an extra tick so the next fresh snapshots include its final writes.
"""

from __future__ import annotations

import copy
import time
import uuid

from huggingface_hub.errors import HfHubHTTPError

ACTIVE = {"claimed", "queued", "running", "unknown", "blocked_dispatch_gap", "settling"}
TERMINAL = {"complete", "failed"}


def _config(config):
    result = {
        "sources": list(dict.fromkeys(config["sources"])),
        "target_documents": config.get("target_documents", 100_000),
        "rounds": config.get("rounds", 3),
        "max_documents": config.get("max_documents", 1000),
        "work_seconds": config.get("work_seconds", 3000),
        "max_campaigns": config.get("max_campaigns", 150),
    }
    if "target_tokens" in config:
        target = config["target_tokens"]
        if type(target) is not int or target < 1:
            raise ValueError("target_tokens must be positive")
        result["target_tokens"] = target
    if "stop_at_token_target" in config:
        if type(config["stop_at_token_target"]) is not bool:
            raise ValueError("stop_at_token_target must be a boolean")
        result["stop_at_token_target"] = config["stop_at_token_target"]
    if "source_rounds" in config:
        overrides = config["source_rounds"]
        if not isinstance(overrides, dict) or any(
            source not in result["sources"] or type(rounds) is not int or not 1 <= rounds <= 6
            for source, rounds in overrides.items()
        ):
            raise ValueError("source_rounds must map configured sources to one through six rounds")
        result["source_rounds"] = dict(overrides)
    if not result["sources"] or any(not isinstance(x, str) or not x for x in result["sources"]):
        raise ValueError("Goal sources must be a nonempty list of source IDs")
    for key, maximum in (("rounds", 3), ("max_documents", 1000), ("work_seconds", 3000)):
        if type(result[key]) is not int or not 1 <= result[key] <= maximum:
            raise ValueError(f"Invalid bounded setting: {key}")
    if type(result["target_documents"]) is not int or result["target_documents"] < 1:
        raise ValueError("target_documents must be positive")
    if type(result["max_campaigns"]) is not int or result["max_campaigns"] < 1:
        raise ValueError("max_campaigns must be positive")
    return result


def cached_token_lower_bound(state, cache, repo_id):
    """Only current immutable shards contribute; partial audits are lower bounds."""
    if cache is None:
        return 0
    if (
        cache.get("schema_version") != 1
        or cache.get("encoding") != "o200k_base"
        or cache.get("repo_id") != repo_id
        or not isinstance(cache.get("shards"), dict)
    ):
        raise ValueError("Incompatible token audit cache")
    total = 0
    seen = set()
    for shard in state["shards"]:
        if shard["path"] in seen:
            raise ValueError("Duplicate source shard path")
        seen.add(shard["path"])
        entry = cache["shards"].get(shard["sha256"])
        if entry is None:
            continue
        if entry.get("path") != shard["path"] or entry.get("rows") != shard["rows"]:
            continue  # A stale/nonmatching cache entry contributes nothing.
        tokens = entry.get("tokens")
        if type(tokens) is not int or tokens < 0:
            raise ValueError("Invalid exact token count")
        total += tokens
    return total


def _source_summary(source_states, sources):
    summaries = {}
    for source in sources:
        snapshot = source_states.get(source)
        if snapshot is None:
            raise ValueError(f"Missing fresh source snapshot: {source}")
        rows = snapshot.get("metrics", {}).get("rows")
        if type(rows) is not int or rows < 0:
            raise ValueError(f"Missing/invalid stored row metric: {source}")
        tokens = snapshot.get("exact_tokens_lower_bound", 0)
        if type(tokens) is not int or tokens < 0:
            raise ValueError(f"Invalid exact token lower bound: {source}")
        remaining = snapshot.get("remaining_supported_candidates")
        if remaining is not None and (type(remaining) is not int or remaining < 0):
            raise ValueError(f"Invalid remaining candidate count: {source}")
        summaries[source] = {
            "metrics": copy.deepcopy(snapshot["metrics"]),
            "exact_tokens_lower_bound": tokens,
            "remaining_supported_candidates": remaining,
            "inventory_complete": snapshot.get("inventory_complete") is True,
            "discovery_complete": snapshot.get("discovery_complete") is True,
            "discovery_pending": snapshot.get("discovery_pending") is True,
            "retry_deferred": snapshot.get("retry_deferred") is True,
            "untracked_active_campaigns": copy.deepcopy(
                snapshot.get("untracked_active_campaigns", [])
            ),
        }
    return summaries


def _reservation(job):
    if job.get("status") in TERMINAL:
        return 0
    # External queued/running campaigns must conservatively reserve the maximum
    # unless their parent supplies the actual, smaller bounded settings.
    reservation = job.get("reserved_documents", 3000)
    if type(reservation) is not int or not 1 <= reservation <= 6000:
        raise ValueError("Job reservations must be integers in [1, 6000]")
    return reservation


def acknowledge_recovery(goal, source, reason, evidence, clock=time.time):
    """Record an operator-verified fix without changing failed job history."""
    if source not in goal["config"]["sources"] or not reason.strip() or not evidence.strip():
        raise ValueError("Recovery requires a configured source, reason and evidence")
    jobs = [job for job in goal["jobs"] if job["source_id"] == source]
    if not jobs or any(job.get("status") not in TERMINAL for job in jobs):
        raise ValueError("Every source campaign must be confirmed terminal before recovery")
    if jobs[-1].get("status") != "failed":
        raise ValueError("Recovery requires the latest source campaign to have failed")
    jobs[-1]["recovery"] = {
        "acknowledged_at_epoch": clock(),
        "reason": reason,
        "evidence": evidence,
    }


def _summary(goal, summaries):
    selected = set(goal["config"]["sources"])
    jobs = [job for job in goal["jobs"] if job["source_id"] in selected]
    rows = sum(source["metrics"]["rows"] for source in summaries.values())
    tokens = sum(source["exact_tokens_lower_bound"] for source in summaries.values())
    token_target = goal["config"].get("target_tokens")
    reserved = sum(_reservation(job) for job in jobs)
    busy = {job["source_id"] for job in jobs if job.get("status") not in TERMINAL}
    failures = dict.fromkeys(selected, 0)
    for job in jobs:
        if job.get("status") == "complete":
            failures[job["source_id"]] = 0
        elif job.get("status") == "failed":
            if job.get("recovery"):
                failures[job["source_id"]] = 0
            else:
                failures[job["source_id"]] += 1
    source_status = {}
    for source, snapshot in summaries.items():
        if snapshot["untracked_active_campaigns"]:
            source_status[source] = "blocked_untracked_campaign"
        elif source in busy:
            source_status[source] = "active_or_unresolved"
        elif failures[source] >= 3:
            source_status[source] = "blocked_repeated_failures"
        elif snapshot["retry_deferred"]:
            source_status[source] = "waiting_retry"
        elif snapshot["discovery_pending"]:
            source_status[source] = "ready"
        elif snapshot["remaining_supported_candidates"] == 0:
            source_status[source] = (
                "scope_exhausted"
                if snapshot["inventory_complete"] and snapshot["discovery_complete"]
                else "coverage_gap"
            )
        elif snapshot["remaining_supported_candidates"] is None:
            source_status[source] = "coverage_gap"
        else:
            source_status[source] = "ready"
    available = max(0, goal["config"]["target_documents"] - rows - reserved)
    if (
        goal["config"].get("stop_at_token_target", True)
        and token_target is not None
        and tokens >= token_target
    ):
        status = "token_target_reached"
    elif rows >= goal["config"]["target_documents"]:
        status = "target_reached"
    elif all(value == "scope_exhausted" for value in source_status.values()):
        status = "selected_scope_exhausted"
    elif len(goal["jobs"]) >= goal["config"]["max_campaigns"]:
        status = "blocked_campaign_limit"
    elif busy:
        status = "collecting"
    elif "ready" in source_status.values():
        status = "ready" if available else "budget_reserved"
    elif any(value.startswith("blocked_") for value in source_status.values()):
        status = "blocked"
    elif "waiting_retry" in source_status.values():
        status = "waiting_retry"
    else:
        status = "coverage_gap"
    return {
        "status": status,
        "stored_rows": rows,
        "exact_tokens_lower_bound": tokens,
        "reserved_documents": reserved,
        "available_document_budget": available,
        "sources": source_status,
    }


def _attach_call(store, job_id, call_id):
    for attempt in range(3):
        state, revision = store.load()
        job = next(job for job in state["collection_goal"]["jobs"] if job["job_id"] == job_id)
        if job.get("call_id") not in (None, call_id):
            raise RuntimeError("Claim already has a different call ID")
        if job.get("status") in TERMINAL:
            raise RuntimeError("Claim became terminal during dispatch")
        job.update(call_id=call_id, status="running")
        try:
            store.commit(state, revision, [], message=f"Attach goal campaign {job_id}")
            return
        except HfHubHTTPError as error:
            if getattr(error.response, "status_code", None) not in (409, 412) or attempt == 2:
                raise


def run_tick(store, source_states, inspect_call, spawn_campaign, goal_config, clock=time.time):
    """Reconcile jobs, reserve budget with CAS, and dispatch at most one per source.

    ``store`` is HFStore for the separate private control repository.
    ``source_states`` maps source IDs to fresh snapshots containing ``metrics.rows``,
    ``remaining_supported_candidates`` and explicit ``inventory_complete``.
    ``inspect_call(call_id)`` returns a string status or dict with ``status`` and
    optional ``result`` and ``child_tree_terminated``. Failed/unknown calls never
    release reservations without explicit child-tree termination confirmation.
    Successful ``complete`` calls are assumed to have a completed child tree;
    an explicitly non-complete campaign result requires termination confirmation.
    ``spawn_campaign`` is called with keyword arguments source_id, campaign_id,
    rounds, max_documents and work_seconds. It returns a call ID or object_id handle.

    A spawn exception or lost attachment leaves a durable unresolved claim. That
    claim is blocked on the next tick, never retried automatically. CAS conflicts
    propagate before dispatch; only attaching an already-spawned call is retried.
    """
    config = _config(goal_config)
    summaries = _source_summary(source_states, config["sources"])
    state, revision = store.load()
    goal = state.setdefault("collection_goal", {})
    if goal.get("config") not in (None, config):
        raise ValueError("Persisted goal config differs; update it explicitly before ticking")
    goal["config"] = config
    jobs = goal.setdefault("jobs", [])
    now = clock()
    tick_id = uuid.uuid4().hex
    goal["last_tick_id"] = tick_id
    for job in jobs:
        if job["source_id"] not in config["sources"] or job.get("status") in TERMINAL:
            continue
        if job.get("status") == "settling":
            # Snapshots were freshly loaded for this later invocation, after final
            # call termination was observed and committed by the previous tick.
            if now > job["terminal_observed_at_epoch"]:
                job["status"] = job["terminal_status"]
            continue
        call_id = job.get("call_id")
        if not call_id:
            job["status"] = "blocked_dispatch_gap"
            continue
        try:
            observation = inspect_call(call_id)
        except Exception as error:  # noqa: BLE001 -- uncertainty preserves the reservation
            observation = {"status": "unknown", "error": type(error).__name__}
        if isinstance(observation, str):
            observation = {"status": observation}
        if not isinstance(observation, dict):
            observation = {"status": "unknown"}
        status = observation.get("status", "unknown")
        result = observation.get("result")
        terminal_safe = observation.get("child_tree_terminated") is True
        if status == "complete" and (
            not isinstance(result, dict) or result.get("status", "complete") == "complete"
        ):
            terminal_safe = True
        job["last_inspected_at_epoch"] = now
        job["last_observation"] = copy.deepcopy(observation)
        if status in {"complete", "failed"} and terminal_safe:
            terminal_status = (
                "complete"
                if status == "complete"
                and (not isinstance(result, dict) or result.get("status", "complete") == "complete")
                else "failed"
            )
            job.update(
                status="settling", terminal_status=terminal_status, terminal_observed_at_epoch=now
            )
        elif status in {"running", "queued"}:
            job["status"] = status
        else:
            job["status"] = "unknown"
    goal["source_snapshots"] = summaries
    goal["updated_at_epoch"] = now
    goal.update(_summary(goal, summaries))
    goal.setdefault("history", []).append(
        {
            "observed_at_epoch": now,
            "stored_rows": goal["stored_rows"],
            "exact_tokens_lower_bound": goal["exact_tokens_lower_bound"],
            "reserved_documents": goal["reserved_documents"],
            "status": goal["status"],
            "source_metrics": {key: value["metrics"] for key, value in summaries.items()},
        }
    )
    store.commit(state, revision, [], message="Reconcile collection goal and source metrics")

    for source in config["sources"]:
        # Reload before each claim: concurrent controller ticks can only win via CAS.
        state, revision = store.load()
        goal = state["collection_goal"]
        if goal["config"] != config:
            raise RuntimeError("Goal config changed during tick")
        if goal.get("last_tick_id") != tick_id:
            # A newer tick owns the freshest metrics; do not allocate against
            # this invocation's older snapshot after another reconciliation.
            break
        summary = _summary(goal, summaries)
        if summary["status"] in {
            "target_reached",
            "token_target_reached",
            "blocked_campaign_limit",
        }:
            break
        if summary["sources"][source] != "ready" or not summary["available_document_budget"]:
            continue
        source_rounds = config.get("source_rounds", {}).get(source, config["rounds"])
        campaign_budget = source_rounds * config["max_documents"]
        budget = min(
            campaign_budget,
            summary["available_document_budget"],
            campaign_budget
            if summaries[source]["discovery_pending"]
            else summaries[source]["remaining_supported_candidates"],
        )
        rounds = min(source_rounds, budget)
        maximum = min(config["max_documents"], budget // rounds)
        job_id = f"goal-{source}-{int(now)}-{uuid.uuid4().hex[:10]}"
        job = {
            "job_id": job_id,
            "source_id": source,
            "campaign_id": job_id,
            "status": "claimed",
            "claimed_at_epoch": clock(),
            "reserved_documents": rounds * maximum,
            "settings": {
                "rounds": rounds,
                "max_documents": maximum,
                "work_seconds": config["work_seconds"],
            },
        }
        goal["jobs"].append(job)
        goal.update(_summary(goal, summaries))
        store.commit(state, revision, [], message=f"Claim goal campaign {job_id}")
        # Never retry dispatch: even an RPC exception can mean the job was launched.
        call = spawn_campaign(source_id=source, campaign_id=job_id, **job["settings"])
        call_id = call if isinstance(call, str) else getattr(call, "object_id", None)
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Spawn must return a nonempty call ID or object_id handle")
        _attach_call(store, job_id, call_id)
    state, _ = store.load()
    return copy.deepcopy(state["collection_goal"])
