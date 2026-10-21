"""Authorized 500M corpus + two 500M students + existing student 4.5M update."""

import json
import logging
import re
import time
from pathlib import Path

import modal

from distillation.modal_app import image as training_image
from distillation.modal_vllm import image as generation_image
from distillation.modal_vllm import volume

app = modal.App("lm-distillation-500m-sweep", include_source=False)
logger = logging.getLogger(__name__)


@app.function(
    image=generation_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=86400,
    retries=0,
    max_containers=4,
    volumes={"/distill": volume},
)
def producer(run_id, worker):
    from distillation.scale_data import generate

    return generate(Path("/distill") / run_id, worker, volume.commit)


@app.function(
    image=training_image,
    gpu="H100",
    cpu=4,
    memory=32768,
    timeout=86400,
    retries=0,
    max_containers=3,
    volumes={"/distill": volume},
)
def student(run_id, name, segment_tokens):
    from distillation.scale_train import run

    return run(Path("/distill") / run_id, name, segment_tokens, volume.commit)


@app.function(
    image=training_image,
    cpu=1,
    memory=2048,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/distill": volume},
)
def supervise(run_id):
    from distillation.scale_data import atomic_json, available_chunks

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    root = Path("/distill") / run_id
    root.mkdir(exist_ok=True)
    plan = {
        "corpus_target_tokens": 500_000_000,
        "producer_workers": 4,
        "students": {
            "fresh-27m": 500_000_000,
            "fresh-48m": 500_000_000,
            "existing-216m": 4_500_000,
        },
        "large_parent": "qwen4b-shared-50m-20261021-01/step-000007162.pt",
        "policy": "No validation-based early stop; finite token targets; durable segment resume",
        "verification": "teacher-generated problems/code/explanations; syntax checked, semantics unverified",
    }
    if (root / "plan.json").exists() and json.loads(
        (root / "plan.json").read_text()
    ) != plan:
        raise ValueError("Plan mismatch")
    atomic_json(root / "plan.json", plan)
    volume.commit()
    calls_file = root / "calls.json"
    prior = json.loads(calls_file.read_text()) if calls_file.exists() else {}
    calls = {key: modal.FunctionCall.from_id(value) for key, value in prior.items()}
    for worker in range(4):
        key = f"producer-{worker}"
        if key not in calls:
            calls[key] = producer.spawn(run_id, worker)
    started = time.monotonic()
    failures = []
    while time.monotonic() - started < 82000:
        volume.reload()
        for key in list(calls):
            try:
                result = calls[key].get(timeout=0)
            except TimeoutError:
                continue
            except Exception as error:
                logger.exception("Sweep child failed: %s", key)
                failures.append({"job": key, "error": str(error)})
                del calls[key]
                continue
            del calls[key]
            if key.startswith("producer-") and result["status"] != "completed":
                if result["status"] == "time_limit_resume_required":
                    calls[key] = producer.spawn(run_id, int(key.rsplit("-", 1)[1]))
                else:
                    failures.append({"job": key, "error": result["status"]})
        chunks = available_chunks(root)
        ready = sum(item[2]["tokens"] for item in chunks)
        progress = {}
        for name, target in plan["students"].items():
            path = root / name / "progress.json"
            progress[name] = (
                json.loads(path.read_text())
                if path.exists()
                else {"tokens": 0, "target": target}
            )
            done = progress[name]["tokens"]
            if (
                name in calls
                or done >= target
                or any(f["job"] == name for f in failures)
            ):
                continue
            if name == "existing-216m":
                # Tokenizers differ: start once 6M frozen-tokenizer tokens are ready;
                # if that yields fewer Qwen targets, another segment resumes later.
                runnable = ready >= 6_000_000 or ready == 500_000_000
                segment = target - done
            else:
                runnable = ready - done >= min(5_000_000, target - done)
                segment = min(10_000_000, ready - done, target - done)
            if runnable:
                calls[name] = student.spawn(run_id, name, segment)
        state = {
            "ready_training_tokens": ready,
            "students": progress,
            "active_jobs": list(calls),
            "failures": failures,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(calls_file, {key: call.object_id for key, call in calls.items()})
        atomic_json(root / "status.json", state)
        volume.commit()
        print(json.dumps(state), flush=True)
        if not calls and (
            failures
            or all(
                progress[name]["tokens"] >= target
                for name, target in plan["students"].items()
            )
        ):
            state["status"] = "failed" if failures else "completed"
            atomic_json(root / "status.json", state)
            volume.commit()
            return state
        time.sleep(30)
    # A bounded continuation keeps the finite run alive beyond one Modal timeout.
    # Producers and students are reattached by persisted call IDs, never duplicated.
    continuation = supervise.spawn(run_id)
    return {"status": "continued", "call_id": continuation.object_id}


@app.local_entrypoint()
def main(run_id: str):
    result = supervise.remote(run_id)
    print(json.dumps(result, indent=2))
