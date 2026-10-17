"""Scheduled, count-bounded continuation using a separate durable HF control repo."""

import json
import os
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
    timeout=14400,
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

    nodes = [node for root in roots for node in flatten(root)]
    terminal = all(node.status.name in {"SUCCESS", "FAILURE", "TERMINATED"} for node in nodes)
    if not terminal:
        return {"status": "running", "child_tree_terminated": False}
    if roots[0].status.name == "SUCCESS":
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


def source_snapshots(mapping, goal, audit_store=None, audit_revision=None):
    from huggingface_hub.errors import EntryNotFoundError

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
        # Older terminal metrics can precede a later inventory expansion. Only
        # the latest run may provide an exact remaining count; otherwise use a hint.
        last = state["runs"][-1] if state["runs"] else {}
        remaining = last.get("remaining_supported_candidates")
        if remaining is None or set(last.get("collection_weights", {})) != set(
            config["collections"]
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
    schedule=modal.Period(minutes=15) if SCHEDULE_ENABLED else None,
)
def tick():
    from goal_controller import run_tick
    from hf_store import HFStore

    config = json.loads(GOAL_CONFIG.read_text())
    mapping = json.loads(GOAL_MAPPING.read_text())
    store = HFStore(config.pop("control_repo"))
    state, control_revision = store.load()
    goal = state.get("collection_goal")
    if not goal:
        raise RuntimeError("Initialize durable control state with all active/queued calls first")
    if goal.get("status") in {"target_reached", "selected_scope_exhausted"}:
        return {"status": goal["status"], "stored_rows": goal.get("stored_rows")}
    result = run_tick(
        store,
        source_snapshots(mapping, goal, store, control_revision),
        inspect_call,
        lambda **kwargs: campaign.spawn(**kwargs),
        config,
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


@app.local_entrypoint()
def main():
    print(json.dumps(tick.remote(), indent=2))
