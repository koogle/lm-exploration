"""Eight-hour teacher campaign, started only after correctness/throughput gates."""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import modal

from core import atomic_json, ignore_source
from modal_app import control_image, image, kernel_image, volume

app = modal.App("lm-qwen35-distillation-teachers", include_source=False)


def execute(run_id, worker, deadline, backend, batch_size, max_batches=0):
    if time.time() >= deadline:
        return {"status": "deadline"}
    proc = subprocess.Popen(
        [
            sys.executable,
            "/experiment/campaign_generate.py",
            f"/runs/{run_id}",
            str(worker),
            str(deadline),
            backend,
            str(batch_size),
            str(max_batches),
        ],
        start_new_session=True,
    )
    try:
        code = proc.wait(timeout=max(1, deadline - time.time()))
        if code:
            raise RuntimeError(f"Teacher exited {code}")
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    volume.reload()
    path = Path("/runs") / run_id / f"producer-{worker}" / "progress.json"
    return (
        json.loads(path.read_text())
        if path.exists()
        else {"status": "deadline_before_output"}
    )


@app.function(
    image=image.env({"HF_HOME": "/tmp/hf-qwen35", "HF_XET_CACHE": "/tmp/xet-qwen35"}),
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=29000,
    retries=0,
    max_containers=256,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def vllm_teacher(run_id, worker, deadline, batch_size, max_batches=0):
    return execute(run_id, worker, deadline, "vllm", batch_size, max_batches)


@app.function(
    image=kernel_image.env(
        {"HF_HOME": "/tmp/hf-qwen35", "HF_XET_CACHE": "/tmp/xet-qwen35"}
    ),
    gpu="H100",
    cpu=4,
    memory=32768,
    timeout=29000,
    retries=0,
    max_containers=256,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def kernel_teacher(run_id, worker, deadline, batch_size, max_batches=0):
    return execute(run_id, worker, deadline, "megakernel", batch_size, max_batches)


archive_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("modal==1.6.2", "huggingface-hub==0.36.0")
    .env(
        {
            "HF_HOME": "/tmp/hf-campaign-archive",
            "HF_XET_CACHE": "/tmp/xet-campaign-archive",
        }
    )
    .add_local_dir(Path(__file__).parent, "/experiment", ignore=ignore_source)
)


@app.function(
    image=archive_image,
    secrets=[modal.Secret.from_name("distillation-huggingface")],
    cpu=2,
    memory=8192,
    timeout=33000,
    retries=2,
    max_containers=1,
    volumes={"/runs": volume},
)
def archive(run_id, workers, deadline):
    from campaign_archive import archive as upload

    return upload(Path("/runs") / run_id, workers, deadline)


@app.function(
    image=control_image,
    cpu=1,
    memory=2048,
    timeout=29200,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def supervise(run_id, config):
    # All thresholds and measured gate evidence are supplied by the qualification launcher.
    if not config["correctness_passed"] or not config["throughput_compared"]:
        raise ValueError("Qualification gates not met")
    if config["backend"] == "megakernel" and config["kernel_speedup"] < 1.1:
        raise ValueError("No material kernel throughput advantage")
    root = Path("/runs") / run_id
    root.mkdir(exist_ok=True, parents=True)
    path = root / "campaign.json"
    if path.exists():
        raise ValueError("Campaign already exists; do not duplicate teachers")
    start = time.time()
    deadline = start + 8 * 3600
    state = {
        **config,
        "started_at": start,
        "deadline": deadline,
        "calls": {},
        "status": "generating",
    }
    atomic_json(path, state)
    volume.commit()
    archiver = archive.spawn(run_id, config["workers"], deadline)
    state["archive_call"] = archiver.object_id
    teacher = kernel_teacher if config["backend"] == "megakernel" else vllm_teacher
    calls = {}
    try:
        while time.time() < deadline:
            for worker in range(config["workers"]):
                key = str(worker)
                if key in calls:
                    try:
                        calls[key].get(timeout=0)
                    except TimeoutError:
                        continue
                    except Exception as error:
                        logging.getLogger(__name__).exception("Teacher failed")
                        state.setdefault("errors", []).append(
                            {"worker": worker, "error": str(error), "at": time.time()}
                        )
                    del calls[key]
                if sum(e["worker"] == worker for e in state.get("errors", [])) >= 2:
                    continue
                if time.time() >= deadline:
                    break
                calls[key] = teacher.spawn(
                    run_id, worker, deadline, config["batch_size"]
                )
                state["calls"] = {k: c.object_id for k, c in calls.items()}
                atomic_json(path, state)
                volume.commit()
            time.sleep(30)
    finally:
        for call in calls.values():
            call.cancel(terminate_containers=True)
        state.update(
            status="generation_window_completed", calls={}, stopped_at=time.time()
        )
        atomic_json(path, state)
        volume.commit()
    return state
