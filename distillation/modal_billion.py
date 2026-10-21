"""Finite 4.5B shared-Qwen-token continuation, supervised across Modal timeouts."""

import json
import logging
import time
from pathlib import Path

import modal

from distillation.modal_app import image
from distillation.modal_vllm import image as generation_image
from distillation.modal_vllm import volume
from distillation.scale_train import FRESH_CORPUS

logger = logging.getLogger(__name__)

app = modal.App("lm-distillation-215m-4p5b", include_source=False)
CORPUS = "python-distillation-500m-20261021-01"
NAME = "shared-qwen-215m-4p5b"
TARGET = 4_500_000_000


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("distillation-huggingface")],
    cpu=2,
    memory=8192,
    timeout=86400,
    retries=2,
    max_containers=1,
    volumes={"/distill": volume},
)
def archive_dataset():
    from distillation.dataset_archive import archive

    return archive(
        Path("/distill") / CORPUS,
        NAME,
        Path("/distill") / FRESH_CORPUS,
        volume.commit,
        volume.reload,
    )


@app.function(
    image=image, cpu=4, memory=8192, timeout=1800, volumes={"/distill": volume}
)
def configure_supply():
    from distillation.dataset_pipeline import freeze_initial_prefix, publish_chunks
    from distillation.scale_data import atomic_json, read_chunk

    root = Path("/distill") / CORPUS
    fresh = Path("/distill") / FRESH_CORPUS
    fresh.mkdir(exist_ok=True)
    plan = freeze_initial_prefix(root, NAME, FRESH_CORPUS)
    supply_path = root / NAME / "supply-plan.json"
    if not supply_path.exists():
        count = sum(
            len(row["qwen_ids"])
            for chunk in plan["initial_chunks"]
            for row in read_chunk(root, chunk)
        )
        atomic_json(
            supply_path,
            {
                "workers": 4,
                "initial_qwen_tokens": count,
                "target": TARGET,
                "headroom": 1.25,
                "buffer_tokens": 100_000_000,
                "sizing": "ceil(student_tps * 1.25 / accepted_teacher_tps)",
            },
        )
    state_path = root / NAME / "supervisor.json"
    state = json.loads(state_path.read_text())
    for call_id in state.get("generation_calls", {}).values():
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
    state["generation_calls"] = {}
    atomic_json(state_path, state)
    publish_chunks(fresh, 4)
    volume.commit()
    return json.loads(supply_path.read_text())


@app.function(
    image=generation_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=86400,
    retries=2,
    max_containers=256,
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
        segment_tokens=10_000_000,
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
    gpu="H100:8",
    cpu=32,
    memory=131072,
    timeout=86400,
    retries=2,
    max_containers=1,
    volumes={"/distill": volume},
)
def train():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            "-m",
            "distillation.scale_train",
            "--root",
            str(Path("/distill") / CORPUS),
            "--name",
            NAME,
            "--segment-tokens",
            "50000000",
        ],
        check=True,
        timeout=80000,
    )
    volume.reload()
    return json.loads((Path("/distill") / CORPUS / NAME / "progress.json").read_text())


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
    from distillation.dataset_pipeline import publish_chunks, teacher_count
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
    archive_call = (
        modal.FunctionCall.from_id(state["archive_call"])
        if state.get("archive_call")
        else None
    )
    started = time.monotonic()
    while time.monotonic() - started < 82000:
        volume.reload()
        supply_path = output / "supply-plan.json"
        supply = json.loads(supply_path.read_text()) if supply_path.exists() else None
        initial_complete = supply is not None or corpus_complete(root)
        ready = (
            supply["initial_qwen_tokens"]
            if supply
            else sum(chunk[2]["tokens"] for chunk in available_chunks(root))
        )
        workers = supply["workers"] if supply else 4
        # Size the pool from actual 8-GPU throughput, after startup has amortized.
        metrics_path = output / "metrics.jsonl"
        if supply and metrics_path.exists():
            metrics = [
                json.loads(line) for line in metrics_path.read_text().splitlines()[-30:]
            ]
            rates = [
                m["segment_tokens_per_second"]
                for m in metrics
                if m.get("world_size") == 8
                and m["tokens"] - state["start_tokens"] >= 5_000_000
            ]
            teacher_rates = []
            for worker in range(workers):
                path = fresh_root / f"producer-{worker}" / "progress.json"
                if path.exists():
                    p = json.loads(path.read_text())
                    if p["generation_seconds"] >= 60:
                        teacher_rates.append(p["qwen_tokens"] / p["generation_seconds"])
            if rates and teacher_rates:
                import statistics

                student_rate = statistics.median(rates[-10:])
                teacher_rate = statistics.median(teacher_rates)
                desired = teacher_count(student_rate, teacher_rate, supply["headroom"])
                workers = max(workers, desired)
                supply.update(
                    workers=workers,
                    measured_student_tps=student_rate,
                    measured_teacher_tps=teacher_rate,
                    desired_teachers=desired,
                )
                atomic_json(supply_path, supply)
        manifest = publish_chunks(fresh_root, workers) if supply else None
        fresh_ready = (
            manifest["qwen_tokens"]
            if manifest
            else sum(chunk[2]["qwen_tokens"] for chunk in available_chunks(fresh_root))
        )
        watermark = ready + (fresh_ready if initial_complete else 0)
        durable = json.loads((output / "progress.json").read_text())
        buffered = watermark - durable["tokens"]
        generation_done = supply is not None and watermark >= TARGET
        if generation_done:
            for producer_call in generation.values():
                producer_call.cancel(terminate_containers=True)
            generation = {}
            manifest["status"] = "completed"
            atomic_json(fresh_root / "manifest.json", manifest)
        if supply:
            if archive_call:
                try:
                    archive_call.get(timeout=0)
                except TimeoutError:
                    pass
                except Exception as error:
                    logger.exception("Dataset archive failed; retrying durable upload")
                    state["archive_error"] = str(error)
                    archive_call = None
                else:
                    archive_call = None
            if archive_call is None:
                archive_call = archive_dataset.spawn()
            state["archive_call"] = archive_call.object_id
        for worker in range(workers):
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
            if generation_done or (supply and buffered >= supply["buffer_tokens"]):
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
            buffer_tokens=buffered,
            teacher_workers=workers,
            teacher_sizing=supply,
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
