"""Finite 4.5B shared-Qwen-token continuation, supervised across Modal timeouts."""

import json
import time
from pathlib import Path

import modal

from distillation.modal_app import image
from distillation.modal_vllm import image as generation_image
from distillation.modal_vllm import volume
from distillation.scale_train import FRESH_CORPUS

app = modal.App("lm-distillation-215m-4p5b", include_source=False)
CORPUS = "python-distillation-500m-20261021-01"
NAME = "shared-qwen-215m-4p5b"
TARGET = 4_500_000_000


@app.function(
    image=generation_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=86400,
    retries=2,
    max_containers=4,
    volumes={"/distill": volume},
)
def fresh_producer(worker):
    from distillation.scale_data import generate

    # Upper bound: consumed old data also counts toward training's 4.5B target.
    # Producers are stopped when training completes; unused tail is not replayed.
    return generate(
        Path("/distill") / FRESH_CORPUS,
        worker,
        volume.commit,
        target=1_125_000_000,
        token_kind="qwen",
        index_offset=100_000_000,
    )


@app.function(
    image=image, cpu=1, memory=2048, timeout=300, volumes={"/distill": volume}
)
def prepare_unique_policy():
    """Run only after the old controller and training call are cancelled."""
    from distillation.scale_data import atomic_json

    output = Path("/distill") / CORPUS / NAME
    path = output / "supervisor.json"
    state = json.loads(path.read_text())
    progress = json.loads((output / "progress.json").read_text())
    if progress.get("corpus_pass", 0) != 0:
        raise ValueError("Cannot claim a fresh-only continuation after replay")
    state.update(
        call_id=None,
        progress=progress,
        drained=False,
        policy="Single pass over fresh generated data; never repeat",
        continuation_corpus=FRESH_CORPUS,
    )
    atomic_json(path, state)
    volume.commit()
    return state


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
            "policy": "Single pass over fresh generated data; never repeat",
        }
    )
    call = modal.FunctionCall.from_id(state["call_id"]) if state["call_id"] else None
    fresh_root = Path("/distill") / FRESH_CORPUS
    generation = {
        key: modal.FunctionCall.from_id(value)
        for key, value in state.get("generation_calls", {}).items()
    }
    state["policy"] = "Single pass over fresh generated data; never repeat"
    started = time.monotonic()
    while time.monotonic() - started < 82000:
        volume.reload()
        initial_complete = corpus_complete(root)
        ready = sum(chunk[2]["tokens"] for chunk in available_chunks(root))
        fresh_ready = sum(
            chunk[2]["qwen_tokens"] for chunk in available_chunks(fresh_root)
        )
        # A byte-stable original prefix must finish before appending fresh shards.
        watermark = ready + (fresh_ready if initial_complete else 0)
        for worker in range(4):
            key = str(worker)
            if key in generation:
                try:
                    result = generation[key].get(timeout=0)
                except TimeoutError:
                    continue
                except Exception as error:
                    state.update(status="failed", error=f"producer {worker}: {error}")
                    atomic_json(state_path, state)
                    volume.commit()
                    raise
                del generation[key]
                if result["status"] == "completed":
                    continue
            path = fresh_root / f"producer-{worker}" / "progress.json"
            if path.exists() and json.loads(path.read_text())["status"] == "completed":
                continue
            generation[key] = fresh_producer.spawn(worker)
            state["generation_calls"] = {k: c.object_id for k, c in generation.items()}
            atomic_json(state_path, state)
            volume.commit()
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
                    for producer_call in generation.values():
                        producer_call.cancel(terminate_containers=True)
                    state["generation_calls"] = {}
                    state["status"] = "completed"
                    atomic_json(state_path, state)
                    volume.commit()
                    return state
                # If the segment filled its 50M budget, more data may already be
                # ready. Otherwise wait for new shards before renting another GPU.
                state["drained"] = result["tokens"] - state["start_tokens"] < 50_000_000
        if call is None and (
            not state.get("drained", False)
            or watermark - state["last_ready"] >= 10_000_000
            or (
                initial_complete
                and corpus_complete(fresh_root)
                and not state.get("final_drain", False)
            )
        ):
            state["start_tokens"] = state.get("progress", {}).get("tokens", 4_500_000)
            state["last_ready"] = watermark
            state["final_drain"] = initial_complete and corpus_complete(fresh_root)
            call = train.spawn()
            state["call_id"] = call.object_id
        state.update(
            status="running" if call else "waiting_for_corpus",
            ready_original_tokenizer_tokens=ready,
            ready_fresh_qwen_tokens=fresh_ready,
            generation_calls={k: c.object_id for k, c in generation.items()},
        )
        atomic_json(state_path, state)
        volume.commit()
        time.sleep(30)
    continuation = supervise.spawn()
    return {"status": "continued", "call_id": continuation.object_id}


@app.local_entrypoint()
def main():
    print(json.dumps(supervise.remote(), indent=2))
