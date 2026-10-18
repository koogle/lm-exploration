"""Bounded remote collection supervisor with durable HF plans and crash-safe claims.

A claim without a terminal worker checkpoint is deliberately not replayed: a lost
RPC response cannot prove that its worker has stopped. Resolve that claim manually.
The caller must enforce a 3,600-second timeout on each remote collection worker.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from huggingface_hub.errors import HfHubHTTPError

from hf_store import COMPONENT, HFStore

ROOT = Path(__file__).resolve().parent
TERMINAL_RUNS = {"bounded_complete", "paused_rate_limit"}
STOPPED = {"complete", "deadline_reached", "ambiguous", "failed", "stopped_worker_failure"}


def run_campaign(
    source_id,
    campaign_id,
    collect_callable,
    rounds=3,
    max_documents=1000,
    work_seconds=3000,
    *,
    after_campaign="",
    predecessor_wait_seconds=1800,
    spawn_callable=None,
    call_loader=None,
    store=None,
    mapping_path=None,
    clock=time.time,
    sleep=time.sleep,
):
    """Run a configured bounded number of remote rounds, preserving claims/results in source HF state.

    ``collect_callable(run_id, max_documents, work_seconds)`` is synchronous and
    should call a separate Modal worker's ``.remote`` method. Alternatively supply
    ``spawn_callable`` returning a handle with ``object_id`` and ``get(timeout=)``
    and ``call_loader(id)`` to reattach durable child calls after supervisor crashes.
    Spawned workers must wait for their persisted call_id before loading HF state.
    Unknown dispatch gaps stop conservatively; dispatch itself is never retried.
    """
    if (spawn_callable is None) != (call_loader is None):
        raise ValueError("spawn_callable and call_loader must be provided together")
    mapping = json.loads(Path(mapping_path or ROOT / "pipeline_sources.json").read_text())
    config = mapping["sources"][source_id]
    limits = mapping["limits"]
    if not COMPONENT.fullmatch(campaign_id):
        raise ValueError("Invalid campaign ID")
    max_rounds = config.get("max_campaign_rounds", 3)
    if type(max_rounds) is not int or not 1 <= max_rounds <= 6:
        raise ValueError("Configured campaign round limit must be one through six")
    if type(rounds) is not int or not 1 <= rounds <= max_rounds:
        raise ValueError("Campaign rounds exceed the source bound")
    if not 1 <= max_documents <= limits["max_documents_per_source"]:
        raise ValueError("Document limit exceeds configured bound")
    if not 1 <= work_seconds <= limits["work_seconds"]:
        raise ValueError("Work duration exceeds configured bound")
    worker_timeout = limits.get("worker_timeout_seconds", 3600)
    if not 1 <= worker_timeout <= 3600 or work_seconds > worker_timeout:
        raise ValueError("Worker timeout must bound work and be at most 3600 seconds")
    store = store or HFStore(config["repo_id"])
    store.ensure(private=mapping["private"])
    invocation_started = clock()
    if after_campaign:
        wait_for_predecessor(
            store,
            after_campaign,
            timeout_seconds=predecessor_wait_seconds,
            clock=clock,
            sleep=sleep,
        )
    settings = {"rounds": rounds, "max_documents": max_documents, "work_seconds": work_seconds}

    if after_campaign:
        settings["after_campaign"] = after_campaign
        settings["predecessor_wait_seconds"] = predecessor_wait_seconds

    def save(state, revision, message):
        store.commit(state, revision, [], message=message)

    def persist_call(index, run_id, call_id):
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Spawned call must expose a nonempty object_id")
        # Retry only metadata CAS conflicts, never dispatch or unrelated failures.
        for attempt in range(3):
            latest, head = store.load()
            entry = latest["campaigns"][campaign_id]["rounds"][index]
            if entry["run_id"] != run_id or entry["status"] != "running":
                raise RuntimeError("Campaign round claim changed during spawn")
            if entry.get("call_id") not in (None, call_id):
                raise RuntimeError("Campaign round already has a different child call")
            entry["call_id"] = call_id
            try:
                save(latest, head, f"Attach campaign child {run_id}")
                return
            except HfHubHTTPError as error:
                status = getattr(error.response, "status_code", None)
                if status not in (409, 412) or attempt == 2:
                    raise

    def record_result(index, run_id, result):
        # Workers mutate this repository. Never commit the pre-call state.
        latest, head = store.load()
        campaign = latest["campaigns"][campaign_id]
        active = campaign["rounds"][index]
        if active["status"] != "running":
            raise RuntimeError("Campaign round claim changed during collection")
        if not isinstance(result, dict) or result.get("run_id") != run_id:
            return False
        if result.get("status") not in TERMINAL_RUNS:
            return False
        active.update(
            status="complete",
            finished_at_epoch=clock(),
            result=copy.deepcopy(result),
            metrics=copy.deepcopy(latest.get("metrics", {})),
        )
        campaign.update(status="running", updated_at_epoch=clock())
        save(latest, head, f"Record campaign round {run_id}")
        return True

    def finish(status):
        latest, head = store.load()
        campaign = latest["campaigns"][campaign_id]
        campaign.update(status=status, updated_at_epoch=clock())
        save(latest, head, f"Campaign {campaign_id}: {status}")
        return copy.deepcopy(latest["campaigns"][campaign_id])

    state, revision = store.load()
    campaigns = state.setdefault("campaigns", {})
    if campaign_id not in campaigns:
        campaigns[campaign_id] = {
            "source_id": source_id,
            "repo_id": config["repo_id"],
            "settings": settings,
            "status": "planned",
            "started_at_epoch": clock(),
            "deadline_epoch": invocation_started
            + max(4, rounds + 1) * 3600
            + (predecessor_wait_seconds if after_campaign else 0),
            "rounds": [
                {"run_id": f"{campaign_id}-round-{i + 1}", "status": "planned"}
                for i in range(rounds)
            ],
        }
        save(state, revision, f"Plan campaign {campaign_id}")
    elif campaigns[campaign_id]["settings"] != settings:
        raise ValueError("Existing campaign settings differ; use a new campaign ID")

    for index in range(rounds):
        while True:
            state, revision = store.load()
            campaign = state["campaigns"][campaign_id]
            if campaign["status"] in STOPPED - {"ambiguous"}:
                return copy.deepcopy(campaign)
            entry = campaign["rounds"][index]
            if entry["status"] == "complete":
                break
            run_id = entry["run_id"]
            matching = [r for r in state["runs"] if r["run_id"] == run_id]
            terminal = matching[-1] if matching else None
            if terminal and terminal["status"] in TERMINAL_RUNS:
                campaign["status"] = "running"
                entry.update(status="complete", result=copy.deepcopy(terminal))
                entry["metrics"] = copy.deepcopy(state.get("metrics", {}))
                entry["finished_at_epoch"] = clock()
                save(state, revision, f"Reconcile campaign round {run_id}")
                if (
                    terminal.get(
                        "remaining_supported_candidates", terminal.get("pending_candidates")
                    )
                    == 0
                ):
                    return finish("complete")
                break
            if entry["status"] == "running" or matching:
                if not entry.get("call_id") or call_loader is None:
                    return finish("ambiguous")
                remaining = campaign["deadline_epoch"] - clock()
                if remaining <= 0:
                    return finish("deadline_reached")
                # Reattach the original call; never launch a replacement worker.
                result = call_loader(entry["call_id"]).get(timeout=remaining)
                if not record_result(index, run_id, result):
                    return finish("ambiguous")
                if (
                    result.get("remaining_supported_candidates", result.get("pending_candidates"))
                    == 0
                ):
                    return finish("complete")
                break
            remaining = campaign["deadline_epoch"] - clock()
            cooldown = state.get("source_retry_after_epoch", 0) - clock()
            # Reserve the full hard timeout, not just cooperative worker duration.
            if remaining < worker_timeout + max(0, cooldown):
                return finish("deadline_reached")
            if cooldown > 0:
                if campaign.get("status") != "waiting_cooldown":
                    campaign.update(status="waiting_cooldown", updated_at_epoch=clock())
                    save(state, revision, f"Campaign {campaign_id}: source cooldown")
                sleep(min(60, cooldown))
                continue
            campaign.update(status="running", updated_at_epoch=clock())
            entry.update(
                status="running",
                started_at_epoch=clock(),
                requires_call_id=spawn_callable is not None,
            )
            save(state, revision, f"Claim campaign round {run_id}")
            # On RPC failure leave claim intact: the remote worker may still run.
            if spawn_callable is None:
                result = collect_callable(run_id, max_documents, work_seconds)
            else:
                call = spawn_callable(run_id, max_documents, work_seconds)
                persist_call(index, run_id, call.object_id)
                remaining = campaign["deadline_epoch"] - clock()
                if remaining <= 0:
                    return finish("deadline_reached")
                result = call.get(timeout=remaining)
            if not record_result(index, run_id, result):
                return finish("ambiguous")
            if result.get("remaining_supported_candidates", result.get("pending_candidates")) == 0:
                return finish("complete")
            break
    return finish("complete")


def wait_for_dispatch(
    source_id,
    run_id,
    *,
    store=None,
    mapping_path=None,
    timeout_seconds=120,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Gate a spawned worker until its supervisor has durably recorded the call ID.

    Call before run_source loads state, to prevent its first checkpoint racing the
    supervisor's call-ID commit. Legacy/direct runs without the flag pass through.
    """
    if not 0 < timeout_seconds <= 120:
        raise ValueError("Dispatch gate timeout must be in (0, 120] seconds")
    if store is None:
        mapping = json.loads(Path(mapping_path or ROOT / "pipeline_sources.json").read_text())
        store = HFStore(mapping["sources"][source_id]["repo_id"])
    deadline = clock() + timeout_seconds
    while True:
        state, _ = store.load()
        entries = [
            entry
            for campaign in state.get("campaigns", {}).values()
            for entry in campaign["rounds"]
            if entry["run_id"] == run_id
        ]
        if len(entries) > 1:
            raise RuntimeError("Multiple campaign claims share this run ID")
        if not entries or not entries[0].get("requires_call_id"):
            return
        entry = entries[0]
        if entry["status"] != "running":
            raise RuntimeError("Spawned worker has no running campaign claim")
        if entry.get("call_id"):
            return
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("Supervisor did not persist child call ID before dispatch deadline")
        sleep(min(2, remaining))


def wait_for_predecessor(
    store, campaign_id, *, timeout_seconds=1800, clock=time.time, sleep=time.sleep
):
    """Bounded read-only wait; only confirmed completion permits writing."""
    if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 10800:
        raise ValueError("Predecessor wait must be 1–10800 seconds")
    deadline = clock() + timeout_seconds
    while True:
        state, _ = store.load()
        predecessor = state.get("campaigns", {}).get(campaign_id)
        if predecessor is None:
            raise ValueError("Predecessor campaign does not exist")
        status = predecessor["status"]
        if status == "complete":
            return
        if status in STOPPED:
            raise RuntimeError(f"Predecessor has not completed safely: {status}")
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("Predecessor did not complete within bounded wait")
        sleep(min(60, remaining))
