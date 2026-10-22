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

sys.path.insert(
    0, str(Path(__file__).parent if modal.is_local() else Path("/experiment"))
)

from core import atomic_json, ignore_source
from modal_app import control_image, image, kernel_image, volume

app = modal.App("lm-qwen35-distillation-teachers")


def execute(run_id, worker, deadline, backend, batch_size, max_batches=0, sweep=False):
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
            "sweep" if sweep else "production",
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
def vllm_teacher(run_id, worker, deadline, batch_size, max_batches=0, sweep=False):
    return execute(run_id, worker, deadline, "vllm", batch_size, max_batches, sweep)


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
def kernel_teacher(run_id, worker, deadline, batch_size, max_batches=0, sweep=False):
    return execute(
        run_id, worker, deadline, "megakernel", batch_size, max_batches, sweep
    )


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
    volume.reload()
    if path.exists():
        state = json.loads(path.read_text())
        if "resume_epoch" in config:
            from campaign_resume import resume_state

            state = resume_state(state, config, time.time())
        elif state.get("status") != "generating":
            return state
    else:
        start = time.time()
        deadline = min(start + 8 * 3600, config["launched_at"] + 23 * 3600)
        if deadline <= start:
            raise TimeoutError("Campaign launch expired while queued")
        state = {
            **config,
            "started_at": start,
            "deadline": deadline,
            "calls": {},
            "status": "generating",
        }
    deadline = state["deadline"]
    atomic_json(path, state)
    volume.commit()
    if not state.get("archive_call") and time.time() < deadline:
        archiver = archive.spawn(run_id, config["workers"], deadline)
        state["archive_call"] = archiver.object_id
        atomic_json(path, state)
        volume.commit()
    teacher = kernel_teacher if config["backend"] == "megakernel" else vllm_teacher
    calls = {
        k: modal.FunctionCall.from_id(c) for k, c in state.get("calls", {}).items()
    }
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
        # An exception or preemption must not cancel healthy teachers. The
        # independently running guardian restores this controller if needed.
        if time.time() >= deadline:
            for call in calls.values():
                call.cancel(terminate_containers=True)
            state.update(
                status="generation_window_completed", calls={}, stopped_at=time.time()
            )
            atomic_json(path, state)
            volume.commit()
    return state


@app.function(
    image=control_image,
    cpu=0.25,
    memory=512,
    timeout=1900,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def qualify_campaign(run_id, deadline):
    """Independent cancellation of both benchmark children, including queued work."""
    if time.time() >= deadline:
        raise TimeoutError("Qualification deadline expired")
    root = Path("/runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    calls = {}
    state = {"deadline": deadline, "calls": {}, "results": {}}
    try:
        for backend, function in [
            ("vllm", vllm_teacher),
            ("megakernel", kernel_teacher),
        ]:
            call = function.spawn(run_id + "-" + backend, 0, deadline, 32, 0, True)
            calls[backend] = call
            state["calls"][backend] = call.object_id
            atomic_json(root / "qualification.json", state)
            volume.commit()
        while calls and time.time() < deadline:
            for backend, call in list(calls.items()):
                try:
                    state["results"][backend] = call.get(timeout=0)
                except TimeoutError:
                    continue
                except Exception as error:
                    logging.getLogger(__name__).exception("Qualification worker failed")
                    state["results"][backend] = {"error": str(error)}
                del calls[backend]
                atomic_json(root / "qualification.json", state)
                volume.commit()
            time.sleep(5)
    finally:
        for call in calls.values():
            call.cancel(terminate_containers=True)
        state["status"] = "deadline" if calls else "completed"
        atomic_json(root / "qualification.json", state)
        volume.commit()
    return state


@app.function(
    image=control_image,
    cpu=0.25,
    memory=512,
    timeout=29200,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def maintain_campaign(run_id, config):
    """Resume controllers after failure; cancel all tracked work at fixed deadline."""
    root = Path("/runs") / run_id
    path = root / "guardian.json"
    deadline = config["resume_deadline"]
    volume.reload()
    guard = json.loads(path.read_text()) if path.exists() else {}
    if guard.get("resume_epoch") != config["resume_epoch"]:
        guard = {
            "resume_epoch": config["resume_epoch"],
            "deadline": deadline,
            "restarts": [],
        }
    call = (
        modal.FunctionCall.from_id(guard["controller_call"])
        if guard.get("controller_call")
        else None
    )
    while time.time() < deadline:
        if call:
            try:
                call.get(timeout=0)
            except TimeoutError:
                time.sleep(30)
                continue
            except Exception as error:
                logging.getLogger(__name__).exception(
                    "Controller failed; restoring saved workers"
                )
                guard["restarts"].append({"at": time.time(), "error": str(error)})
            call = None
        if time.time() >= deadline:
            break
        call = supervise.spawn(run_id, config)
        guard["controller_call"] = call.object_id
        atomic_json(path, guard)
        volume.commit()
        time.sleep(30)
    # Stop the controller first, then cancel its persisted children, including queued inputs.
    if call:
        call.cancel(terminate_containers=True)
    volume.reload()
    state = json.loads((root / "campaign.json").read_text())
    for child in state.get("calls", {}).values():
        modal.FunctionCall.from_id(child).cancel(terminate_containers=True)
    state.update(status="generation_window_completed", calls={}, stopped_at=time.time())
    atomic_json(root / "campaign.json", state)
    guard["status"] = "deadline_completed"
    atomic_json(path, guard)
    volume.commit()
    return guard
