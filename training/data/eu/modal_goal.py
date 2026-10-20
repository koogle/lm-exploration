"""Scheduled, count-bounded continuation using a separate durable HF control repo."""

import importlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
GOAL_MAPPING = Path("/root/eu/goal_pipeline_sources.json")
GOAL_CONFIG = Path("/root/eu/collection_goal.json")
APP_NAME = os.environ.get("EU_MODAL_APP_NAME", "lm-exploration-eu-100k")
SCHEDULE_ENABLED = os.environ.get("EU_GOAL_SCHEDULE", "1") == "1"
REMOTE_ENV = {"EU_MODAL_APP_NAME": APP_NAME, "EU_GOAL_SCHEDULE": "1" if SCHEDULE_ENABLED else "0"}
app = modal.App(APP_NAME)
secret = modal.Secret.from_name("eu-corpus-huggingface", required_keys=["HF_TOKEN"])
image = None
if modal.is_local():
    # Build shared tooling locally; the remote module must not import an unmounted
    # build-time entrypoint or read local requirements/CA paths during hydration.
    from modal_pipeline import image as base_image

    image = base_image
    for name in ("goal_controller.py", "goal_pipeline_sources.json", "collection_goal.json"):
        image = image.add_local_file(ROOT / name, f"/root/eu/{name}")


@app.function(
    image=image,
    env=REMOTE_ENV,
    secrets=[secret],
    cpu=1,
    memory=2048,
    timeout=3600,
    max_containers=4,
    retries=0,
)
def collect(source_id: str, run_id: str, max_documents: int, work_seconds: int):
    from campaign import wait_for_dispatch
    from pipeline import run_source

    wait_for_dispatch(source_id, run_id, mapping_path=GOAL_MAPPING)
    return run_source(source_id, run_id, max_documents, work_seconds, mapping_path=GOAL_MAPPING)


@app.function(
    image=image,
    env=REMOTE_ENV,
    secrets=[secret],
    cpu=0.125,
    memory=512,
    timeout=28800,
    max_containers=4,
    retries=0,
)
def campaign(source_id: str, campaign_id: str, rounds: int, max_documents: int, work_seconds: int):
    from campaign import run_campaign

    return run_campaign(
        source_id,
        campaign_id,
        None,
        rounds,
        max_documents,
        work_seconds,
        spawn_callable=lambda run_id, maximum, seconds: collect.spawn(
            source_id, run_id, maximum, seconds
        ),
        call_loader=modal.FunctionCall.from_id,
        mapping_path=GOAL_MAPPING,
    )


def inspect_call(call_id):
    """Never classify a failed supervisor as safe while a descendant is active."""
    call = modal.FunctionCall.from_id(call_id)
    roots = call.get_call_graph()
    if not roots:
        return {"status": "unknown", "child_tree_terminated": False}

    def flatten(node):
        yield node
        for child in node.children:
            yield from flatten(child)

    # Modal may return the shared ancestor and sibling campaigns too. Only the
    # requested call and its descendants determine whether this writer stopped.
    matches = [node for root in roots for node in flatten(root) if node.function_call_id == call_id]
    if len(matches) != 1:
        return {"status": "unknown", "child_tree_terminated": False}
    target = matches[0]
    nodes = list(flatten(target))
    terminal = all(node.status.name in {"SUCCESS", "FAILURE", "TERMINATED"} for node in nodes)
    if not terminal:
        return {"status": "running", "child_tree_terminated": False}
    if target.status.name == "SUCCESS":
        try:
            result = call.get(timeout=0)
        except Exception as error:  # noqa: BLE001 - preserve uncertainty after terminal graph check
            return {
                "status": "failed",
                "child_tree_terminated": True,
                "error_type": type(error).__name__,
            }
        status = result.get("status") if isinstance(result, dict) else None
        return {
            "status": "complete" if status == "complete" else "failed",
            "child_tree_terminated": True,
            "result": {"status": status},
        }
    return {"status": "failed", "child_tree_terminated": True}


def retry_deferred(state, timestamp=None):
    """Avoid empty campaigns while all remaining work is waiting to retry."""
    timestamp = time.time() if timestamp is None else timestamp
    if state.get("source_retry_after_epoch", 0) > timestamp:
        return True
    last = state.get("runs", [])[-1] if state.get("runs") else {}
    if (
        last.get("status") != "bounded_complete"
        or last.get("attempted") != 0
        or not last.get("remaining_supported_candidates")
        or not last.get("finished_at")
    ):
        return False
    # A zero-attempt round cannot justify another immediate campaign. Wait for
    # the earliest persisted retry, with a ten-minute floor to bound write churn.
    finished = datetime.fromisoformat(last["finished_at"]).timestamp()
    future = [
        entry["retry_after_epoch"]
        for entry in state.get("failures", {}).values()
        if entry.get("attempts", 0) < 3 and entry.get("retry_after_epoch", 0) > timestamp
    ]
    return timestamp < max(finished + 600, min(future) if future else 0)


def source_snapshots(mapping, goal, audit_store=None, audit_revision=None):
    from huggingface_hub.errors import EntryNotFoundError

    from eu_corpus import digest
    from goal_controller import cached_token_lower_bound
    from hf_store import HFStore
    from pipeline import inventory_expansion_needed

    tracked = {job.get("campaign_id", job["job_id"]) for job in goal.get("jobs", [])}
    result = {}
    for source in goal["config"]["sources"]:
        config = mapping["sources"][source]
        state, revision = HFStore(config["repo_id"]).load()
        token_cache = None
        if audit_store is not None and goal["config"].get("target_tokens") is not None:
            if not audit_revision:
                raise ValueError("Pinned control revision required for token cache reads")
            try:
                token_cache = json.loads(
                    audit_store._download_bytes(f"tokens/{source}.json", audit_revision)
                )
            except EntryNotFoundError:
                pass
        tokens = cached_token_lower_bound(state, token_cache, config["repo_id"])
        collections = state["collections"]
        global_limit = mapping.get("limits", {}).get("inventory_limit_per_collection", 100000)
        desired_limit = min(
            config.get("inventory_limit_per_collection", global_limit), global_limit
        )
        discovery_pending = any(
            name not in collections or inventory_expansion_needed(collections[name], desired_limit)
            for name in config["collections"]
        )
        # Saved eligibility is valid only for the adapter and collections that
        # measured it. Repairs can make previously exhausted URLs runnable again.
        last = state["runs"][-1] if state["runs"] else {}
        remaining = last.get("remaining_supported_candidates")
        adapter = importlib.import_module(config["adapter"])
        adapter_sha256 = digest(Path(adapter.__file__).read_bytes())
        if (
            remaining is None
            or last.get("adapter_sha256") != adapter_sha256
            or set(last.get("collection_weights", {})) != set(config["collections"])
        ):
            # Conservative upper bound, only a runnable hint until a worker measures it.
            remaining = max(
                0,
                sum(c.get("candidate_count", 0) for c in collections.values())
                - len(state["completed"]),
            )
        result[source] = {
            "metrics": state["metrics"],
            "exact_tokens_lower_bound": tokens,
            "revision": revision,
            "remaining_supported_candidates": remaining,
            "discovery_pending": discovery_pending,
            "retry_deferred": not discovery_pending and retry_deferred(state),
            "inventory_complete": not discovery_pending
            and all(
                collections[name].get("enumeration", {}).get("complete") is True
                for name in config["collections"]
            ),
            "discovery_complete": False,
            "untracked_active_campaigns": [
                name
                for name, entry in state.get("campaigns", {}).items()
                if name not in tracked
                and entry.get("status") in {"planned", "running", "waiting_cooldown", "ambiguous"}
            ],
        }
    return result


@app.function(
    image=image,
    env=REMOTE_ENV,
    secrets=[secret],
    cpu=0.125,
    memory=1024,
    timeout=600,
    max_containers=1,
    retries=0,
    schedule=modal.Period(minutes=1) if SCHEDULE_ENABLED else None,
)
def tick():
    from hf_store import HFStore

    config = json.loads(GOAL_CONFIG.read_text())
    mapping = json.loads(GOAL_MAPPING.read_text())
    store = HFStore(config.pop("control_repo"))
    result = reconcile_tick(
        store, mapping, config, inspect_call, lambda **kwargs: campaign.spawn(**kwargs)
    )
    return {
        key: result.get(key)
        for key in (
            "status",
            "stored_rows",
            "exact_tokens_lower_bound",
            "reserved_documents",
            "available_document_budget",
            "sources",
        )
    }


def reconcile_tick(store, mapping, config, inspect, spawn):
    """Settle terminal calls using a second fresh snapshot in the same invocation."""
    from goal_controller import run_tick

    for _ in range(2):
        state, revision = store.load()
        goal = state.get("collection_goal")
        if not goal:
            raise RuntimeError(
                "Initialize durable control state with all active/queued calls first"
            )
        result = run_tick(
            store, source_snapshots(mapping, goal, store, revision), inspect, spawn, config
        )
        if not any(
            job.get("status") == "settling" and job["source_id"] in config["sources"]
            for job in result["jobs"]
        ):
            break
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(tick.remote(), indent=2))
