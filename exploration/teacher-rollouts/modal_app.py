"""A single bounded H100 allocation; no autoscaling, retries or student training."""

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent if modal.is_local() else Path("/experiment")
sys.path.insert(0, str(HERE))

from core import atomic_json, ignore_source, supervise_call, validate_config

CONFIG = json.loads((HERE / "config.json").read_text())
app = modal.App("lm-exploration-teacher-rollouts")


image = (
    modal.Image.from_registry(CONFIG["image"])
    .entrypoint([])
    # The pinned vLLM runtime provides python3 but no unversioned python.
    # Modal's pip overlay and our worker use the unversioned command.
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python")
    .pip_install_from_requirements(str(HERE / "requirements.lock"))
    .add_local_dir(
        HERE,
        "/experiment",
        ignore=ignore_source,
    )
)
KERNEL_IMAGE = "nvidia/cuda@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7"
kernel_image = (
    modal.Image.from_registry(KERNEL_IMAGE, add_python="3.12")
    .pip_install_from_requirements(str(HERE / "megakernel/requirements.lock"))
    .add_local_dir(
        HERE,
        "/experiment",
        copy=True,
        ignore=ignore_source,
    )
    .run_commands(
        "nvcc -std=c++17 -O3 -arch=sm_90 -DTARGET_SM=90 --shared -Xcompiler=-fPIC /experiment/megakernel/kernel.cu -o /experiment/megakernel/kernel.so"
    )
)
BACKEND = os.environ.get("ROLLOUT_BACKEND", "baseline")
if BACKEND == "kernel-sweep":
    kernel_image = kernel_image.run_commands(
        "python /experiment/megakernel/build_sweep.py /experiment/megakernel/builds"
    )
image = (kernel_image if BACKEND in ("megakernel", "kernel-sweep") else image).env(
    {"ROLLOUT_BACKEND": BACKEND}
)
volume = modal.Volume.from_name(
    "lm-exploration-teacher-rollouts", create_if_missing=True
)


@app.function(
    image=image,
    gpu="H100",
    cpu=(4, 4),
    memory=(32768, 32768),
    timeout=1200,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def worker(run_id, attempt, manifest, deadline, supervisor_id):
    return execute_worker(
        run_id,
        attempt,
        manifest,
        deadline,
        supervisor_id,
        {
            "megakernel": "megakernel/probe.py",
            "kernel-sweep": "megakernel/sweep.py",
        }.get(manifest.get("backend"), "runner.py"),
    )


def execute_worker(run_id, attempt, manifest, deadline, supervisor_id, script):
    c = manifest["config"]
    validate_config(c)
    if not c["budget_authorized"] or not c["authorization_reference"]:
        raise ValueError("budget not authorized")
    for name in (run_id, attempt):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            raise ValueError("invalid ID")
    remaining = min(c["runtime_seconds"], int(deadline - time.time()))
    if remaining < 60:
        raise TimeoutError("Insufficient time before launch deadline")
    root = Path("/runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"input-{attempt}.json"
    if path.exists():
        raise ValueError("attempt already used")
    atomic_json(path, manifest)
    atomic_json(
        root / f"supervision-{attempt}.json",
        {
            "deadline_unix": deadline,
            "supervisor_call_id": supervisor_id,
            "worker_call_id": modal.current_function_call_id(),
        },
    )
    volume.commit()
    started = time.monotonic()
    status = "failed"
    try:
        # Parent timeout includes startup, model loading, generation and verification.
        proc = subprocess.Popen(
            ["python", "/experiment/" + script, str(root), attempt, str(path)],
            cwd="/experiment",
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=remaining - 45)
        except subprocess.TimeoutExpired:
            status = "timeout"
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        if returncode:
            raise RuntimeError(f"runner failed with exit code {returncode}")
        status = "completed"
    finally:
        atomic_json(
            root / f"worker-{attempt}.json",
            {"elapsed_seconds": time.monotonic() - started, "status": status},
        )
        volume.commit()
    return json.loads((root / "attempts" / attempt / "report.json").read_text())


# A separate, small CPU function remains alive while the GPU call queues/runs.
control_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"ROLLOUT_BACKEND": BACKEND})
    .pip_install_from_requirements(str(HERE / "requirements.lock"))
    .add_local_dir(
        HERE,
        "/experiment",
        ignore=ignore_source,
    )
)
supervision = modal.Dict.from_name(
    "lm-exploration-teacher-rollout-deadlines", create_if_missing=True
)


@app.function(
    image=control_image,
    cpu=(0.125, 0.125),
    memory=(256, 256),
    timeout=1320,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def supervisor(run_id, attempt, manifest, deadline):
    c = manifest["config"]
    validate_config(c)
    remaining = min(c["runtime_seconds"], int(deadline - time.time()) - 30)
    if remaining < 60 or deadline > time.time() + c["runtime_seconds"] + 60:
        raise TimeoutError("Launch deadline expired or exceeds permitted horizon")
    supervisor_id = modal.current_function_call_id()
    key = f"{run_id}/{attempt}"
    state = {
        "deadline_unix": deadline,
        "supervisor_call_id": supervisor_id,
        "worker_call_id": None,
        "status": "spawning",
        "effective_worker_timeout": remaining,
    }
    supervision[key] = state
    call = worker.with_options(timeout=remaining).spawn(
        run_id, attempt, manifest, deadline, supervisor_id
    )
    try:
        state.update(worker_call_id=call.object_id, status="running")
        supervision[key] = state
        report = supervise_call(call, deadline)
        state["status"] = "completed"
        report["supervision"] = state
        return report
    finally:
        if state["status"] != "completed":
            # Also covers failure to persist the ID before entering supervise_call.
            modal.FunctionCall.from_id(call.object_id).cancel(terminate_containers=True)
            state["status"] = "cancelled_or_failed"
        supervision[key] = state
