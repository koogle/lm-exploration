"""Frozen-checkpoint follow-up; one H100, two-hour absolute deadline, no training."""

import json
import time
import traceback
from pathlib import Path

import modal

from .concept_modal import data, probe_image, registry, runs

app = modal.App("kda-concept-routes")


@app.function(image=probe_image, timeout=7200, retries=0)
def supervisor(run, deadline):
    registry[run + "/ready"] = True
    while time.time() < deadline:
        if registry.get(run + "/done", False):
            return
        time.sleep(min(10, max(0, deadline - time.time())))
    for cid in registry.get(run + "/calls", []):
        modal.FunctionCall.from_id(cid).cancel()
    registry[run + "/expired"] = True


@app.function(
    image=probe_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=6900,
    retries=0,
    volumes={"/runs": runs, "/data": data},
)
def worker(run, deadline, provenance):
    while not registry.get(run + "/registered", False):
        if time.time() > deadline:
            raise TimeoutError("Registration deadline")
        time.sleep(1)
    if time.time() > deadline - 600:
        raise TimeoutError("Launch deadline")
    from model_hybrid.concept_routes import run as probe

    folder = Path("/runs") / run
    folder.mkdir(exist_ok=True, parents=True)
    try:
        result = probe(folder, runs, deadline, provenance)
    except Exception as error:  # noqa: BLE001
        result = {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
    (folder / "completion.json").write_text(json.dumps(result, indent=2))
    runs.commit()
    registry[run + "/result"] = result
    registry[run + "/done"] = True
    return result
