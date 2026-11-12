"""Detection runs on one Modal GPU, each stopped within an hour of launch.

Run from post-training/detection/:

    modal run --detach modal_app.py::main --run-id moondream2-001
    modal run --detach modal_app.py::decoders --run-id decoders-001
    modal run --detach modal_app.py::photon --run-id moondream31-001 \\
        --config configs/moondream3.1-photon.json

    modal run --detach modal_app.py::slots --run-id slots-001

`main` evaluates a config (default `configs/moondream2.json`) on an L4. `slots` trains
the slot head of `configs/slots.json` on an H100 within three hours of launch. `photon`
evaluates a config through Moondream's Photon engine on an L40S. `decoders` runs `configs/decoders.json`
once per decoder (PyTorch eager, this folder's loop, the custom CUDA kernels,
torch.compile) on the same images and reports how each agrees with eager.

Each launch is one GPU call with `retries=0`, writing predictions image by image and
summaries to the `lm-exploration-parallel-detection` volume under
`evaluations/<run id>/`. If Modal preempts the container, the rescheduled call
continues from the recorded images inside the same deadline. An independent CPU
supervisor cancels the call if it is still queued or running at the launch deadline.
The local command waits for the result and saves it under `artifacts/<run id>/`.
"""

import json
import subprocess
import time
from pathlib import Path

import modal

LAUNCH_LIMIT_SECONDS = 60 * 60
TRAIN_LIMIT_SECONDS = 3 * 60 * 60
COMMIT_EVERY_IMAGES = 50

HERE = Path(__file__).parent
if modal.is_local():
    PACKAGE = HERE / "src" / "parallel_detection"
    DEADLINE = HERE.parents[1] / "training" / "workflows" / "deadline.py"
else:
    PACKAGE = Path("/root/parallel_detection")
    DEADLINE = Path("/root/workflows/deadline.py")

app = modal.App("lm-exploration-parallel-detection")
volume = modal.Volume.from_name("lm-exploration-parallel-detection", create_if_missing=True)
# The repository's shared deadline helpers, used by both functions.
control_image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    DEADLINE, "/root/workflows/deadline.py"
)
PACKAGES = (
    "torch==2.6.0",
    "transformers==4.52.4",
    "accelerate==1.7.0",
    "einops==0.8.1",
    "pyvips-binary==8.16.0",
    "pyvips==2.2.3",
    "numpy==2.2.6",
    "scipy==1.15.3",
    "pillow==11.2.1",
    "pyarrow==20.0.0",
    "huggingface-hub==0.32.4",
)
# The CUDA development image exploration/teacher-rollouts builds its kernels with.
CUDA_IMAGE = "nvidia/cuda@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7"


def with_source(base):
    return (
        base.pip_install(*PACKAGES)
        .env({"HF_HOME": "/runs/hf-cache"})
        .add_local_file(DEADLINE, "/root/workflows/deadline.py")
        .add_local_dir(PACKAGE, "/root/parallel_detection")
    )


image = with_source(modal.Image.debian_slim(python_version="3.12"))
# Photon (the `moondream` package) brings its own PyTorch and precompiled kernels.
photon_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "moondream==2.6.1",
        *[p for p in PACKAGES if p.startswith(("numpy", "scipy", "pillow", "pyarrow"))],
    )
    .env({"HF_HOME": "/runs/hf-cache"})
    .add_local_file(DEADLINE, "/root/workflows/deadline.py")
    .add_local_dir(PACKAGE, "/root/parallel_detection")
)
kernel_image = with_source(modal.Image.from_registry(CUDA_IMAGE, add_python="3.12"))


def run_evaluation(run_id, config, provenance, deadline):
    from parallel_detection.run import evaluate as run

    out, saved = start(run_id, deadline)
    if (out / "summary.json").exists():
        raise ValueError(f"Run {run_id} already has results")
    try:
        return run(config, out, "cuda", provenance, saved)
    finally:
        volume.commit()


def start(run_id, deadline):
    """Refuse late work and return the run folder and the per-image volume commit."""
    if time.time() >= deadline:
        raise RuntimeError("Launch deadline reached; refusing new work")
    out = Path("/runs/evaluations") / run_id

    def saved(images):
        if images % COMMIT_EVERY_IMAGES == 0:
            volume.commit()

    return out, saved


@app.function(
    image=image,
    gpu="L4",
    timeout=LAUNCH_LIMIT_SECONDS,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def evaluate(run_id, config, provenance, deadline):
    return run_evaluation(run_id, config, provenance, deadline)


@app.function(
    image=photon_image,
    gpu="L40S",
    timeout=LAUNCH_LIMIT_SECONDS,
    retries=0,
    max_containers=2,
    volumes={"/runs": volume},
)
def evaluate_photon(run_id, config, provenance, deadline):
    return run_evaluation(run_id, config, provenance, deadline)


@app.function(
    image=kernel_image,
    gpu="L4",
    timeout=LAUNCH_LIMIT_SECONDS,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def compare_decoders(run_id, config, provenance, deadline):
    """Evaluate the same tasks with each decoder; a decoder that fails records its error."""
    import gc
    import traceback

    import torch

    from parallel_detection.metrics import agreement
    from parallel_detection.run import evaluate as run
    from parallel_detection.run import read_records

    out, saved = start(run_id, deadline)
    if (out / "comparison.json").exists():
        raise ValueError(f"Run {run_id} already has results")
    comparison = {"provenance": provenance, "decoders": {}}
    try:
        for decoder in config["decoders"]:
            folder = out / decoder
            try:
                if (folder / "summary.json").exists():
                    summary = json.loads((folder / "summary.json").read_text())
                else:
                    summary = run(
                        config,
                        folder,
                        "cuda",
                        provenance,
                        saved,
                        decoder=decoder,
                        kernel_library="/tmp/decode_step.so",
                    )
                reference = read_records(out / config["decoders"][0] / "predictions.jsonl")
                comparison["decoders"][decoder] = {
                    "datasets": summary["datasets"],
                    "environment": summary["environment"],
                    "agreement_with_first": agreement(
                        reference, read_records(folder / "predictions.jsonl")
                    ),
                }
            except Exception:  # noqa: BLE001 - the failure is the recorded result
                comparison["decoders"][decoder] = {"error": traceback.format_exc()[-4000:]}
            gc.collect()
            torch.cuda.empty_cache()
        (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        return comparison
    finally:
        volume.commit()


@app.function(
    image=image,
    gpu="H100",
    memory=(81920, 81920),
    timeout=TRAIN_LIMIT_SECONDS,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def train_slots(run_id, config, provenance, deadline):
    """Encode the configured COCO shards once, then train the slot head on them."""
    from huggingface_hub import hf_hub_download

    from parallel_detection.train_slots import encode_shard, train

    if time.time() >= deadline:
        raise RuntimeError("Launch deadline reached; refusing new work")
    out = Path("/runs/slots") / run_id
    if (out / "report.json").exists():
        raise ValueError(f"Run {run_id} already has results")
    data = config["data"]
    folder = Path("/runs/slots") / f"features-{config['min_area']}-{config['max_objects']}"
    stems = []
    for name in data["files"]:
        stem = folder / Path(name).stem
        if not stem.with_suffix(".json").exists():
            parquet = hf_hub_download(
                data["repo"], name, repo_type=data["repo_type"], revision=data["revision"]
            )
            encode_shard(parquet, stem, config)
            volume.commit()
        stems.append(stem)
    # Stop training early enough to write the checkpoint and report.
    report = train(config, stems, out, saved=volume.commit, deadline=deadline - 900)
    report["provenance"] = provenance
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.function(
    image=control_image, cpu=0.25, memory=512, timeout=TRAIN_LIMIT_SECONDS + 600, retries=0
)
def enforce_deadline(call_ids, deadline):
    """Independent supervisor: cancel every call of this launch still running at its deadline."""
    from workflows.deadline import calls_finished, cancel_at_deadline

    cancelled = cancel_at_deadline(
        deadline, lambda: call_ids, lambda: calls_finished(call_ids), poll_seconds=60
    )
    return {"stage": "completed"} if cancelled is None else {"cancelled": cancelled}


def launch(function, run_id, config, result_name, gpu="L4", limit=LAUNCH_LIMIT_SECONDS):
    deadline = time.time() + limit
    git = ["git", "-C", str(HERE)]
    provenance = {
        "run_id": run_id,
        "git_commit": subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output([*git, "status", "--porcelain", "."], text=True)),
        "gpu": gpu,
    }
    remaining = int(deadline - time.time())
    call = function.with_options(timeout=remaining).spawn(
        run_id, json.loads(Path(config).read_text()), provenance, deadline
    )
    supervisor = enforce_deadline.spawn([call.object_id], deadline)
    out = HERE / "artifacts" / run_id
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "call": call.object_id,
        "supervisor": supervisor.object_id,
        "deadline_unix": deadline,
        "results": f"lm-exploration-parallel-detection/evaluations/{run_id}/",
    }
    (out / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    result = call.get(timeout=limit)
    (out / result_name).write_text(json.dumps(result, indent=2) + "\n")


@app.local_entrypoint()
def main(run_id: str, config: str = "configs/moondream2.json"):
    launch(evaluate, run_id, config, "summary.json")


@app.local_entrypoint()
def decoders(run_id: str, config: str = "configs/decoders.json"):
    launch(compare_decoders, run_id, config, "comparison.json")


@app.local_entrypoint()
def photon(run_id: str, config: str = "configs/moondream3.1-photon.json"):
    launch(evaluate_photon, run_id, config, "summary.json", gpu="L40S")


@app.local_entrypoint()
def slots(run_id: str, config: str = "configs/slots.json"):
    launch(train_slots, run_id, config, "report.json", gpu="H100", limit=TRAIN_LIMIT_SECONDS)
