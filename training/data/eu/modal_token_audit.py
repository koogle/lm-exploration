"""Bounded token measurement, isolated from document collection workers."""

import json
from pathlib import Path

import modal

app = modal.App("lm-exploration-eu-token-audit")
image = None
if modal.is_local():
    from modal_pipeline import image as base_image

    image = base_image
    for name in ("token_audit.py", "audit_schedule.py", "goal_pipeline_sources.json"):
        image = image.add_local_file(Path(__file__).with_name(name), f"/root/eu/{name}")


@app.function(image=image, cpu=1, memory=512, timeout=120, retries=0)
def verify_tokenizer():
    import tiktoken

    encoder = tiktoken.get_encoding("o200k_base")
    return {
        "tokenizer": encoder.name,
        "sample_tokens": len(encoder.encode("EU corpus verification")),
    }


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("eu-corpus-huggingface", required_keys=["HF_TOKEN"])],
    cpu=1,
    memory=2048,
    timeout=1800,
    max_containers=1,
    retries=0,
)
def audit(source_id: str, max_shards: int = 1000, work_seconds: int = 1500):
    from token_audit import audit_source

    return audit_source(source_id, max_shards=max_shards, work_seconds=work_seconds)


@app.function(image=image, cpu=0.125, memory=256, timeout=4000, max_containers=1, retries=0)
def resume_after(not_before_epoch: float):
    """Honor a bounded Hub cooldown without keeping a local session open."""
    import time

    if not 0 <= not_before_epoch - time.time() <= 3600:
        raise ValueError("Resume must be scheduled within the next hour")
    while time.time() < not_before_epoch:
        time.sleep(max(0, min(60, not_before_epoch - time.time())))
    # Each audit retains its own hard runtime bound and existing durable cache.
    return [
        {"source_id": source, "call_id": audit.spawn(source, 3000, 1500).object_id}
        for source in ("ecb", "ema")
    ]


def inspect_audit(call_id):
    call = modal.FunctionCall.from_id(call_id)
    roots = call.get_call_graph()
    if not roots:
        return {"status": "unknown"}
    if any(root.status.name not in {"SUCCESS", "FAILURE", "TERMINATED"} for root in roots):
        return {"status": "running"}
    try:
        call.get(timeout=5)
        return {"status": "complete"}
    except TimeoutError:
        return {"status": "unknown"}
    except Exception as error:  # noqa: BLE001 - remote terminal failures have diverse types
        response = getattr(error, "response", None)
        result = {"status": "failed", "http_status": getattr(response, "status_code", None)}
        if result["http_status"] == 429:
            header = getattr(response, "headers", {}).get("retry-after", "3600")
            try:
                result["retry_after_seconds"] = max(3600, float(header))
            except ValueError:
                result["retry_after_seconds"] = 3600
        return result


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("eu-corpus-huggingface", required_keys=["HF_TOKEN"])],
    cpu=0.125,
    memory=1024,
    timeout=600,
    max_containers=1,
    retries=0,
    schedule=modal.Period(hours=1),
)
def scheduled_tick():
    from audit_schedule import run_tick
    from hf_store import HFStore
    from token_audit import AuditCache

    store = HFStore("jfrick/eu-corpus-control")
    state, _ = store.load()
    return run_tick(
        AuditCache(store, "schedule"),
        state["collection_goal"]["status"],
        lambda source: audit.spawn(source, 3000, 1500),
        inspect_audit,
        max_pairs=48,
    )


@app.local_entrypoint()
def main(source: str = "ecb", max_shards: int = 1000, work_seconds: int = 1500):
    print(json.dumps(audit.remote(source, max_shards, work_seconds), indent=2))
