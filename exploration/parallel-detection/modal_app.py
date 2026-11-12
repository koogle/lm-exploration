"""Detection baseline on one Modal GPU, stopped within an hour of launch.

Run from exploration/parallel-detection/:

    modal run modal_app.py --run-id baseline-001

One L4 call with `retries=0` evaluates `configs/baseline.json`, writing predictions
image by image and the summary to the `lm-exploration-parallel-detection` volume under
`evaluations/<run id>/`. An independent CPU supervisor cancels the call if it is still
queued or running at the launch deadline. The local command waits for the summary and
saves it under `artifacts/<run id>/`.
"""

import json
import subprocess
import time
from pathlib import Path

import modal

LAUNCH_LIMIT_SECONDS = 60 * 60
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
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
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
    .env({"HF_HOME": "/runs/hf-cache"})
    .add_local_file(DEADLINE, "/root/workflows/deadline.py")
    .add_local_dir(PACKAGE, "/root/parallel_detection")
)


@app.function(
    image=image,
    gpu="L4",
    timeout=LAUNCH_LIMIT_SECONDS,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def evaluate(run_id, config, provenance, deadline):
    from parallel_detection.run import evaluate as run

    if time.time() >= deadline:
        raise RuntimeError("Launch deadline reached; refusing new work")
    out = Path("/runs/evaluations") / run_id
    if out.exists():
        raise ValueError(f"Run {run_id} already has results")

    def saved(images):
        if images % COMMIT_EVERY_IMAGES == 0:
            volume.commit()

    try:
        return run(config, out, "cuda", provenance, saved)
    finally:
        volume.commit()


@app.function(
    image=control_image, cpu=0.25, memory=512, timeout=LAUNCH_LIMIT_SECONDS + 600, retries=0
)
def enforce_deadline(call_ids, deadline):
    """Independent supervisor: cancel every call of this launch still running at its deadline."""
    from workflows.deadline import calls_finished, cancel_at_deadline

    cancelled = cancel_at_deadline(
        deadline, lambda: call_ids, lambda: calls_finished(call_ids), poll_seconds=60
    )
    return {"stage": "completed"} if cancelled is None else {"cancelled": cancelled}


@app.local_entrypoint()
def main(run_id: str, config: str = "configs/baseline.json"):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    git = ["git", "-C", str(HERE)]
    provenance = {
        "run_id": run_id,
        "git_commit": subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output([*git, "status", "--porcelain", "."], text=True)),
        "gpu": "L4",
    }
    remaining = int(deadline - time.time())
    call = evaluate.with_options(timeout=remaining).spawn(
        run_id, json.loads(Path(config).read_text()), provenance, deadline
    )
    supervisor = enforce_deadline.spawn([call.object_id], deadline)
    out = HERE / "artifacts" / run_id
    out.mkdir(parents=True, exist_ok=True)
    launch = {
        "call": call.object_id,
        "supervisor": supervisor.object_id,
        "deadline_unix": deadline,
        "results": f"lm-exploration-parallel-detection/evaluations/{run_id}/",
    }
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps(launch, indent=2))
    summary = call.get(timeout=LAUNCH_LIMIT_SECONDS)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["datasets"], indent=2))
