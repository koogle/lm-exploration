"""Finite 4.5B shared-Qwen-token continuation, supervised across Modal timeouts."""

import json
import time
from pathlib import Path

import modal

from distillation.modal_app import image
from distillation.modal_vllm import volume

app = modal.App("lm-distillation-215m-4p5b", include_source=False)
CORPUS = "python-distillation-500m-20261021-01"
NAME = "shared-qwen-215m-4p5b"
TARGET = 4_500_000_000


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    timeout=86400,
    retries=2,
    max_containers=1,
    volumes={"/distill": volume},
)
def train():
    from distillation.scale_train import run

    return run(Path("/distill") / CORPUS, NAME, 50_000_000, volume.commit)


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=86400,
    retries=2,
    max_containers=1,
    volumes={"/distill": volume},
)
def supervise():
    from distillation.scale_data import atomic_json, available_chunks
    from distillation.scale_train import corpus_complete

    root = Path("/distill") / CORPUS
    output = root / NAME
    output.mkdir(exist_ok=True)
    state_path = output / "supervisor.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "last_ready": 0,
            "call_id": None,
            "target": TARGET,
            "corpus": CORPUS,
            "policy": "No validation early stopping; repeat only completed corpus",
        }
    )
    call = modal.FunctionCall.from_id(state["call_id"]) if state["call_id"] else None
    started = time.monotonic()
    while time.monotonic() - started < 82000:
        volume.reload()
        ready = sum(chunk[2]["tokens"] for chunk in available_chunks(root))
        if call:
            try:
                result = call.get(timeout=0)
            except TimeoutError:
                pass
            except Exception as error:
                state.update(status="failed", error=str(error))
                atomic_json(state_path, state)
                volume.commit()
                raise
            else:
                state.update(progress=result, call_id=None)
                call = None
                if result["tokens"] == TARGET:
                    state["status"] = "completed"
                    atomic_json(state_path, state)
                    volume.commit()
                    return state
                # If the segment filled its 50M budget, more data may already be
                # ready. Otherwise wait for new shards before renting another GPU.
                state["drained"] = result["tokens"] - state["start_tokens"] < 50_000_000
        if call is None and (
            not state.get("drained", False)
            or ready - state["last_ready"] >= 10_000_000
            or corpus_complete(root)
        ):
            state["start_tokens"] = state.get("progress", {}).get("tokens", 4_500_000)
            state["last_ready"] = ready
            call = train.spawn()
            state["call_id"] = call.object_id
        state.update(
            status="running" if call else "waiting_for_corpus",
            ready_original_tokenizer_tokens=ready,
        )
        atomic_json(state_path, state)
        volume.commit()
        time.sleep(30)
    continuation = supervise.spawn()
    return {"status": "continued", "call_id": continuation.object_id}


@app.local_entrypoint()
def main():
    print(json.dumps(supervise.remote(), indent=2))
