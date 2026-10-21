"""Private ECB/EMA continuation on eight H100s, stacked on optimization PR #17.

Deploying allocates no GPUs. Preparation is an independently spawned CPU job.
Starting the GPU workflow requires its completed, immutable snapshot.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

from distributed.modal_train import data_volume, run_volume, worker

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "numpy", "pydantic>=2.11,<3", "tokenizers==0.22.2")
    .pip_install("huggingface_hub==0.36.0", "pyarrow==19.0.1")
    .env(
        {
            "PYTHONPATH": "/project",
            "TOKENIZERS_PARALLELISM": "true",
            "RAYON_NUM_THREADS": "8",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    .add_local_dir("100m", "/project/model_100m")
    .add_local_dir("10m", "/project/model_10m")
    .add_local_dir("trainer", "/project/trainer")
    .add_local_dir("distributed", "/project/distributed")
    .add_local_dir("workflows", "/project/workflows")
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer", "/project/tokenizer")
    .add_local_dir("tests", "/project/tests")
    .add_local_file("generate.py", "/project/generate.py")
)
app = modal.App("simple-llm-eu-continuation")
PARENT = "simple-llm-124m-8k-5b-20261018-s03/step-000064227.pt"


def snapshot_path(snapshot):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/snapshots/[a-zA-Z0-9_-]+", snapshot):
        raise ValueError("Invalid snapshot path")
    return Path("/data/scholarly-code") / snapshot


@app.function(
    image=image,
    cpu=16,
    memory=32768,
    timeout=14400,
    retries=0,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/data": data_volume},
)
def prepare(snapshot, datasets):
    from huggingface_hub.errors import HfHubHTTPError

    from trainer.eu_continuation import prepare_snapshot

    folder = snapshot_path(snapshot)
    try:
        for attempt in range(8):
            try:
                return prepare_snapshot(folder, datasets, commit=data_volume.commit)
            except HfHubHTTPError as exc:
                if exc.response is None or exc.response.status_code not in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    raise
                data_volume.commit()
                if attempt == 7:
                    raise RuntimeError(
                        "HF download retries exhausted; snapshot progress retained"
                    ) from None
                print(
                    json.dumps(
                        {
                            "stage": "download_retry",
                            "http_status": exc.response.status_code,
                            "attempt": attempt + 1,
                        }
                    ),
                    flush=True,
                )
                # Respect the HF five-minute quota window without losing cursors.
                for _ in range(10):
                    time.sleep(30)
    finally:
        data_volume.commit()


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=300,
    retries=0,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def inspect(snapshot="", run_id=""):
    result = {}
    if snapshot:
        path = snapshot_path(snapshot)
        for name in ("manifest.json", "identity.json"):
            if (path / name).exists():
                result[name] = json.loads((path / name).read_text())
    if run_id:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
            raise ValueError("Invalid run ID")
        folder = Path("/runs") / run_id
        for name in ("workflow.json", "results.json", "report.json", "latest.json"):
            if (folder / name).exists():
                result[name] = json.loads((folder / name).read_text())
        if (folder / "metrics.jsonl").exists():
            result["recent_metrics"] = [
                json.loads(x) for x in (folder / "metrics.jsonl").read_text().splitlines()[-5:]
            ]
    return result


@app.function(
    image=image,
    cpu=4,
    memory=12288,
    timeout=1800,
    retries=0,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def initialize(run_id, snapshot, settings):
    import torch

    from distributed.config import DistributedConfig
    from distributed.modal_train import validate_paths
    from trainer.eu_continuation import migrate_checkpoint

    torch.set_num_threads(4)
    validate_paths(run_id, snapshot, PARENT)
    folder = Path("/runs") / (run_id + "-seed")
    lineage = migrate_checkpoint(
        Path("/runs") / PARENT,
        folder / "step-000000000.pt",
        snapshot_path(snapshot) / "manifest.json",
        DistributedConfig(**settings),
    )
    (folder / "lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")
    run_volume.commit()
    return {"seed": str(folder.relative_to("/runs") / "step-000000000.pt"), "lineage": lineage}


@app.function(
    image=image,
    gpu="H100:8",
    cpu=32,
    memory=98304,
    timeout=14400,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def segment(run_id, snapshot, settings, provenance, resume, max_steps=None, qualify=False):
    import torch

    if torch.cuda.device_count() != 8:
        raise RuntimeError("Expected eight GPUs")
    if qualify:
        folder = Path("/runs") / (run_id + "-qualification")
        folder.mkdir(exist_ok=False)
        result = subprocess.run(
            [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                "/project/tests/check_optimization_cuda.py",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
            env={**os.environ, "TORCHINDUCTOR_COMPILE_THREADS": "1"},
        )
        (folder / "cuda.log").write_text(result.stdout + result.stderr)
        run_volume.commit()
        if result.returncode:
            raise RuntimeError("Eight-GPU correctness gate failed: " + result.stderr[-1500:])
    return worker.local(
        run_id,
        snapshot,
        settings,
        provenance,
        resume,
        10800,
        max_steps,
        dataset_root="scholarly-code",
    )


def mark(folder, stage, **details):
    from trainer.loop import atomic_json

    atomic_json(folder / "workflow.json", {"stage": stage, "updated_unix": time.time(), **details})
    run_volume.commit()


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=1800,
    retries=0,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def assessment(run_id, snapshot, checkpoint, label):
    from trainer.eu_evaluation import evaluate

    folder = Path("/runs") / run_id
    result = evaluate(
        Path("/runs") / checkpoint,
        snapshot_path(snapshot) / "manifest.json",
        Path(
            "/data/scholarly-code/scholarly-small-5b-raw-20261017/snapshots/"
            "scholarly-modern-10m-5b-20261017/manifest.json"
        ),
        folder / f"evaluation-{label}.json",
    )
    run_volume.commit()
    return result


@app.function(
    image=image, cpu=1, memory=2048, timeout=18000, retries=0, volumes={"/runs": run_volume}
)
def after_preparation(run_id, snapshot, prepare_call_id, provenance):
    """Durably queue training behind CPU prep; no idle GPU allocation."""
    folder = Path("/runs") / (run_id + "-request")
    folder.mkdir(exist_ok=False)
    mark(folder, "preparing", snapshot=snapshot, prepare_call_id=prepare_call_id)
    try:
        prepared = modal.FunctionCall.from_id(prepare_call_id).get(timeout=14400)
        count = prepared["sources"]["eu"]["splits"]["train"]["loss_tokens"]
        call = start.spawn(run_id, snapshot, provenance)
        mark(
            folder,
            "controller_started",
            controller_call_id=call.object_id,
            target_tokens=count,
            snapshot=snapshot,
        )
        return {"controller_call_id": call.object_id, "target_tokens": count}
    except Exception as exc:
        mark(folder, "failed", error=f"{type(exc).__name__}: {str(exc)[:500]}")
        raise


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=43200,
    retries=0,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def start(run_id, snapshot, provenance, *, compile_model=True):
    """One pass, preserving Adam; qualify, restore, then bounded continuation."""
    from distributed.modal_train import validate_paths

    validate_paths(run_id, snapshot, "")
    manifest = json.loads((snapshot_path(snapshot) / "manifest.json").read_text())
    target = manifest["sources"]["eu"]["splits"]["train"]["loss_tokens"]
    if manifest.get("continuation_passes") != 1 or not 1_000_000 < target <= 5_000_000_000:
        raise ValueError("Expected a completed, bounded one-pass EU snapshot")
    folder = Path("/runs") / run_id
    folder.mkdir(exist_ok=False)
    settings = {
        "optimizer": "adamw",
        "data_loader": "raw",
        "world_size": 8,
        "global_windows": 32,
        "microbatch": 2,
        "context": 8192,
        "target_tokens": target,
        "warmup_tokens": 1,
        "peak_lr": 3e-5,
        "min_lr_ratio": 1.0,
        "checkpoint_every": 500,
        "evaluate_every": 250,
        "allow_unreviewed_data": True,
        "prefetch_batches": 4,
        "crop_padding": True,
        "cache_rope": True,
        "compile_model": compile_model,
        "n_layer": 14,
        "n_head": 12,
        "n_embd": 768,
    }
    (folder / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    mark(folder, "initializing", snapshot=snapshot, target_tokens=target)
    previous = None
    try:
        seed = initialize.remote(run_id, snapshot, settings)
        resume = seed["seed"]
        (folder / "lineage.json").write_text(json.dumps(seed["lineage"], indent=2) + "\n")
        assessment.remote(run_id, snapshot, PARENT, "before")
        for index in range(5):
            segment_id = f"{run_id}-s{index:02d}"
            mark(
                folder,
                "qualifying" if index < 2 else "training",
                segment=segment_id,
                target_tokens=target,
                phase_tokens=previous["total_loss_tokens"] if previous else 0,
                parent_tokens=seed["lineage"]["parent_tokens"],
                snapshot=snapshot,
            )
            report = segment.remote(
                segment_id,
                snapshot,
                settings,
                provenance,
                resume,
                2 if index < 2 else None,
                qualify=index == 0,
            )
            if previous and previous["final_validation"] != report["initial_validation"]:
                raise RuntimeError("Fresh-process restore validation mismatch")
            if report.get("data_error") or report["stop_reason"] not in (
                "step_limit",
                "time_limit",
                "token_limit",
                "source_two_pass_limit",
            ):
                raise RuntimeError("Training failed or stopped unexpectedly")
            if previous and report["total_loss_tokens"] <= previous["total_loss_tokens"]:
                raise RuntimeError("Continuation made no progress")
            (folder / f"segment-{index:02d}.json").write_text(json.dumps(report, indent=2) + "\n")
            if (
                target - report["total_loss_tokens"]
                < settings["global_windows"] * settings["context"]
            ):
                mark(
                    folder,
                    "completed",
                    target_tokens=target,
                    phase_tokens=report["total_loss_tokens"],
                    lifetime_tokens=report["lifetime_loss_tokens"],
                    unconsumed_tail_tokens=target - report["total_loss_tokens"],
                    final_checkpoint=report["final_checkpoint"],
                )
                final = str(Path(report["final_checkpoint"]).relative_to("/runs"))
                call = assessment.spawn(run_id, snapshot, final, "after")
                (folder / "assessment-call.json").write_text(
                    json.dumps({"call_id": call.object_id}) + "\n"
                )
                run_volume.commit()
                return report
            previous = report
            resume = str(Path(report["final_checkpoint"]).relative_to("/runs"))
        raise RuntimeError("Reached bounded segment limit")
    except Exception as exc:
        mark(
            folder,
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            phase_tokens=previous["total_loss_tokens"] if previous else 0,
        )
        raise
