"""Bounded, isolated L40S/H100 throughput comparison; no production restart."""

import logging
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent if modal.is_local() else Path("/experiment")))

from campaign_modal import execute
from core import atomic_json
from modal_app import control_image, image, volume

app = modal.App("lm-qwen35-hardware-benchmark")


def measure(run_id, deadline, batch_size):
    import subprocess

    from audit_targets import audit

    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    started = time.time()
    state = execute(run_id, 0, deadline, "vllm", batch_size, 256 // batch_size)
    metrics = state.get("batch_metrics", [])
    seconds = sum(x["durable_seconds"] for x in metrics)
    result = {
        "hardware": hardware,
        "batch_size": batch_size,
        "wall_seconds_including_startup": time.time() - started,
        "durable_seconds": seconds,
        "batches": metrics,
        "generated_tokens": state.get("generated_tokens", 0),
        "accepted_tokens": state.get("accepted_tokens", 0),
        "complete": state.get("records") == 256,
    }
    if seconds:
        result["generated_tokens_per_second"] = result["generated_tokens"] / seconds
        result["accepted_tokens_per_second"] = result["accepted_tokens"] / seconds
    root = Path("/runs") / run_id
    chunks = state.get("chunks", [])
    if chunks and time.time() < deadline:
        result["payload_audit"] = audit(root / "producer-0" / chunks[0]["file"])
    atomic_json(root / "hardware-result.json", result)
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=1200,
    retries=0,
    max_containers=2,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def l40s(run_id, deadline, batch_size):
    return measure(run_id, deadline, batch_size)


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=1200,
    retries=0,
    max_containers=2,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def h100(run_id, deadline, batch_size):
    return measure(run_id, deadline, batch_size)


@app.function(
    image=control_image,
    cpu=0.25,
    memory=512,
    timeout=1260,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def supervise(run_id, deadline):
    root = Path("/runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    state = {"deadline": deadline, "calls": {}, "results": {}}
    calls = {}
    try:
        for gpu, fn in [("l40s", l40s), ("h100", h100)]:
            for batch in [32, 128]:
                if time.time() >= deadline:
                    break
                name = f"{gpu}-b{batch}"
                call = fn.spawn(run_id + "-" + name, deadline, batch)
                calls[name] = call
                state["calls"][name] = call.object_id
                atomic_json(root / "benchmark.json", state)
                volume.commit()
        while calls and time.time() < deadline:
            for name, call in list(calls.items()):
                try:
                    state["results"][name] = call.get(timeout=0)
                except TimeoutError:
                    continue
                except Exception as error:
                    logging.getLogger(__name__).exception("Hardware benchmark failed")
                    state["results"][name] = {"error": str(error)}
                del calls[name]
                atomic_json(root / "benchmark.json", state)
                volume.commit()
            time.sleep(5)
    finally:
        for call in calls.values():
            call.cancel(terminate_containers=True)
        state["status"] = "deadline" if calls else "completed"
        atomic_json(root / "benchmark.json", state)
        volume.commit()
    return state
