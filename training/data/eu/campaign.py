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

from hf_store import COMPONENT, HFStore

ROOT = Path(__file__).resolve().parent
TERMINAL_RUNS = {"bounded_complete", "paused_rate_limit"}
STOPPED = {"complete", "deadline_reached", "ambiguous", "failed"}


def run_campaign(
    source_id,
    campaign_id,
    collect_callable,
    rounds=3,
    max_documents=1000,
    work_seconds=3000,
    *,
    store=None,
    mapping_path=None,
    clock=time.time,
    sleep=time.sleep,
):
    """Run at most three remote rounds, preserving claims/results in source HF state.

    ``collect_callable(run_id, max_documents, work_seconds)`` is synchronous and
    should call a separate Modal worker's ``.remote`` method. CAS conflicts abort;
    the supervisor never overwrites concurrent worker progress or retries dispatch.
    """
    mapping = json.loads(Path(mapping_path or ROOT / "pipeline_sources.json").read_text())
    config = mapping["sources"][source_id]
    limits = mapping["limits"]
    if not COMPONENT.fullmatch(campaign_id):
        raise ValueError("Invalid campaign ID")
    if not 1 <= rounds <= 3:
        raise ValueError("Campaigns allow one to three rounds")
    if not 1 <= max_documents <= limits["max_documents_per_source"]:
        raise ValueError("Document limit exceeds configured bound")
    if not 1 <= work_seconds <= limits["work_seconds"]:
        raise ValueError("Work duration exceeds configured bound")
    worker_timeout = limits.get("worker_timeout_seconds", 3600)
    if not 1 <= worker_timeout <= 3600 or work_seconds > worker_timeout:
        raise ValueError("Worker timeout must bound work and be at most 3600 seconds")
    store = store or HFStore(config["repo_id"])
    store.ensure(private=mapping["private"])
    settings = {"rounds": rounds, "max_documents": max_documents, "work_seconds": work_seconds}

    def save(state, revision, message):
        store.commit(state, revision, [], message=message)

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
            "deadline_epoch": clock() + 4 * 3600,
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
            if campaign["status"] in STOPPED:
                return copy.deepcopy(campaign)
            entry = campaign["rounds"][index]
            if entry["status"] == "complete":
                break
            run_id = entry["run_id"]
            matching = [r for r in state["runs"] if r["run_id"] == run_id]
            terminal = matching[-1] if matching else None
            if terminal and terminal["status"] in TERMINAL_RUNS:
                entry.update(status="complete", result=copy.deepcopy(terminal))
                entry["metrics"] = copy.deepcopy(state.get("metrics", {}))
                entry["finished_at_epoch"] = clock()
                save(state, revision, f"Reconcile campaign round {run_id}")
                if terminal.get("pending_candidates") == 0:
                    return finish("complete")
                break
            if entry["status"] == "running" or matching:
                return finish("ambiguous")
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
            entry.update(status="running", started_at_epoch=clock())
            save(state, revision, f"Claim campaign round {run_id}")
            # On RPC failure leave claim intact: the remote worker may still run.
            result = collect_callable(run_id, max_documents, work_seconds)
            # Workers mutate this repository. Never commit the pre-call state.
            latest, head = store.load()
            active = latest["campaigns"][campaign_id]["rounds"][index]
            if active["status"] != "running":
                raise RuntimeError("Campaign round claim changed during collection")
            if not isinstance(result, dict) or result.get("run_id") != run_id:
                return finish("ambiguous")
            if result.get("status") not in TERMINAL_RUNS:
                return finish("ambiguous")
            active.update(
                status="complete",
                finished_at_epoch=clock(),
                result=copy.deepcopy(result),
                metrics=copy.deepcopy(latest.get("metrics", {})),
            )
            save(latest, head, f"Record campaign round {run_id}")
            if result.get("pending_candidates") == 0:
                return finish("complete")
            break
    return finish("complete")
