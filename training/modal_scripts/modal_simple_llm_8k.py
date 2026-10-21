"""Durable eight-H100 / 8K / 5B run with asynchronous Hub inference checkpoints.

Deploying this module does not start training. Explicitly spawn start(run_id).
"""

import json
import time
from pathlib import Path

import modal

from distributed.image import project_image

image = (
    project_image("huggingface-hub==1.33.0", "safetensors>=0.4", env={"RAYON_NUM_THREADS": "8"})
    .add_local_dir("publication", "/project/publication")
    .add_local_file(
        "workflows/model_configs/historical/h100x8-8k-5b.json", "/project/settings.json"
    )
)
app = modal.App("simple-llm-124m-8k-5b")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
run_volume = modal.Volume.from_name("lm-exploration-training")
REPO = "jfrick/scholarly-code-modern-gpt-124m-step2"
SNAPSHOT = "scholarly-small-5b-raw-20261017/snapshots/scholarly-modern-10m-5b-20261017"


def record(run_id, stage, **details):
    from trainer.loop import atomic_json

    folder = Path("/runs") / run_id
    folder.mkdir(parents=True, exist_ok=True)
    status = dict(run_id=run_id, stage=stage, updated_unix=time.time(), **details)
    atomic_json(folder / "workflow.json", status)
    run_volume.commit()
    print(json.dumps(status), flush=True)


@app.function(
    image=image,
    gpu="H100:8",
    cpu=32,
    memory=65536,
    timeout=25200,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def train_segment(run_id, snapshot, settings, provenance, resume="", max_steps=None):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from distributed.modal_train import worker

    if torch.cuda.device_count() != 8:
        raise RuntimeError("Expected eight visible GPUs")
    # Fail before a long run if this device cannot execute the flash SDPA path.
    q = torch.randn(1, 12, 8192, 64, device="cuda", dtype=torch.bfloat16)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    torch.cuda.synchronize()
    del q
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "visible_gpus": 8,
                "flash_sdpa_8k_probe": "passed",
                "devices": [torch.cuda.get_device_name(i) for i in range(8)],
            }
        ),
        flush=True,
    )
    return worker.local(
        run_id,
        snapshot,
        settings,
        provenance,
        resume,
        21600,
        max_steps,
        dataset_root="scholarly-code",
    )


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=86400,
    retries=0,
    max_containers=1,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": run_volume},
)
def publish_progress(run_id):
    import shutil

    from huggingface_hub import HfApi

    from publication.export import export_checkpoint
    from trainer.loop import atomic_json

    api = HfApi()
    folder = Path("/runs") / run_id
    prior = folder / "hub.json"
    run_volume.reload()
    published = json.loads(prior.read_text()).get("step", -1) if prior.exists() else -1
    started = time.monotonic()
    while time.monotonic() - started < 79200:
        run_volume.reload()
        status = json.loads((folder / "workflow.json").read_text())
        candidates = []
        for latest in Path("/runs").glob(f"{run_id}-s*/latest.json"):
            value = json.loads(latest.read_text())
            candidates.append((value["step"], latest.parent / value["checkpoint"]))
        if candidates:
            step, checkpoint = max(candidates)
            if step > published:
                try:
                    output = Path("/tmp/hub-export")
                    shutil.rmtree(output, ignore_errors=True)
                    exported = export_checkpoint(
                        checkpoint,
                        output,
                        checkpoint.parent / "tokenizer/tokenizer.json",
                        data_manifest=checkpoint.parent / "data-manifest.json",
                    )
                    # The authored model card is maintained separately; never replace it here.
                    info = api.upload_folder(
                        repo_id=REPO,
                        folder_path=str(output),
                        ignore_patterns=["README.md"],
                        commit_message=f"Simple LLM 124M 8K: step {step}, {exported['tokens']} tokens",
                    )
                    progress = {
                        "step": exported["step"],
                        "tokens": exported["tokens"],
                        "checkpoint": str(checkpoint),
                        "repo_id": REPO,
                        "revision": info.oid,
                        "updated_unix": time.time(),
                    }
                    atomic_json(prior, progress)
                    run_volume.commit()
                    published = step
                    print(json.dumps(progress), flush=True)
                except Exception as exc:  # noqa: BLE001 -- retry uploads without stopping GPU training
                    atomic_json(
                        folder / "hub-error.json",
                        {
                            "step": step,
                            "error": f"{type(exc).__name__}: {exc}",
                            "updated_unix": time.time(),
                        },
                    )
                    run_volume.commit()
                    print(f"Hub upload failed at step {step}: {type(exc).__name__}", flush=True)
                    time.sleep(30)
                    continue
        if status["stage"] in ("completed", "failed") and (
            not candidates or published >= max(candidates)[0]
        ):
            return {"published_step": published, "training_stage": status["stage"]}
        time.sleep(30)
    call = publish_progress.spawn(run_id)
    return {"continuation_call_id": call.object_id, "published_step": published}


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/runs": run_volume},
)
def continue_training(run_id, settings, provenance, previous=None, first_segment=0):
    from workflows.checks import check_progress, check_resume

    resume = str(Path(previous["final_checkpoint"]).relative_to("/runs")) if previous else ""
    try:
        for segment in range(first_segment, min(first_segment + 3, 24)):
            segment_id = f"{run_id}-s{segment:02d}"
            record(
                run_id,
                "qualifying" if segment < 2 else "training",
                segment=segment_id,
                completed_tokens=previous["total_loss_tokens"] if previous else 0,
                target_tokens=settings["target_tokens"],
                snapshot=SNAPSHOT,
                repo_id=REPO,
            )
            report = train_segment.remote(
                segment_id, SNAPSHOT, settings, provenance, resume, 2 if segment < 2 else None
            )
            if previous:
                check_resume(previous, report)
            done = check_progress(
                report,
                previous["total_loss_tokens"] if previous else 0,
                settings["target_tokens"],
                settings["global_windows"] * settings["context"],
            )
            run_volume.reload()
            (Path("/runs") / run_id / f"segment-{segment:02d}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            if segment == 1:
                call = publish_progress.spawn(run_id)
                record(
                    run_id,
                    "qualified",
                    publisher_call_id=call.object_id,
                    completed_tokens=report["total_loss_tokens"],
                    repo_id=REPO,
                )
            if done:
                record(
                    run_id,
                    "completed",
                    trained_tokens=report["total_loss_tokens"],
                    final_checkpoint=report["final_checkpoint"],
                    repo_id=REPO,
                )
                return report
            previous = report
            resume = str(Path(report["final_checkpoint"]).relative_to("/runs"))
        if segment + 1 >= 24:
            raise RuntimeError("Reached bounded segment limit")
        record(run_id, "continuing", completed_tokens=previous["total_loss_tokens"], resume=resume)
        call = continue_training.spawn(run_id, settings, provenance, previous, segment + 1)
        return {"continuation_call_id": call.object_id}
    except Exception as exc:
        record(run_id, "failed", error=f"{type(exc).__name__}: {exc}", resume=resume)
        raise


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=300,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def start(run_id, provenance):
    from distributed.config import DistributedConfig
    from distributed.modal_train import validate_paths

    validate_paths(run_id, SNAPSHOT, "")
    folder = Path("/runs") / run_id
    if folder.exists():
        raise ValueError("Run ID already exists")
    config = DistributedConfig(**json.loads(Path("/project/settings.json").read_text()))
    manifest = Path("/data/scholarly-code") / SNAPSHOT / "manifest.json"
    data = json.loads(manifest.read_text())
    capacity = min(
        2 * s["splits"]["train"]["loss_tokens"] * 100 / s["weight_percent"]
        for s in data["sources"].values()
    )
    if capacity < config.target_tokens:
        raise ValueError("Existing snapshot lacks five-billion-token mixture capacity")
    record(
        run_id,
        "queued",
        target_tokens=config.target_tokens,
        settings=config.as_dict(),
        snapshot=SNAPSHOT,
        repo_id=REPO,
        provenance=provenance,
    )
    call = continue_training.spawn(run_id, config.as_dict(), provenance)
    return {"training_call_id": call.object_id, "snapshot": SNAPSHOT, "capacity": capacity}
