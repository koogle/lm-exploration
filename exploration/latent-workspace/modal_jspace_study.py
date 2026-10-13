"""Bounded matched augmentation workers; authoritative resumable controller."""

import json
from pathlib import Path

import modal

from latent_workspace.budget import reserve

app = modal.App("latent-workspace-jspace-study")
cache = modal.Volume.from_name("latent-workspace-qwen-cache")
runs = modal.Volume.from_name("latent-workspace-qwen-runs")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv==0.10.12")
    .add_local_file("pyproject.toml", "/project/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/project/uv.lock", copy=True)
    .run_commands("cd /project && uv sync --frozen --no-install-project")
    .env({"PATH": "/project/.venv/bin:/usr/local/bin:/usr/bin:/bin", "HF_HOME": "/cache/hf"})
    .add_local_file("jspace_next.py", "/project/jspace_next.py", copy=True)
    .add_local_file("jspace_study.py", "/project/jspace_study.py", copy=True)
    .add_local_python_source("latent_workspace")
)


@app.function(
    image=image,
    cpu=8,
    memory=131072,
    timeout=3600,
    retries=0,
    single_use_containers=True,
    volumes={"/cache": cache, "/runs": runs},
)
def holdout(protocol, run_id):
    import sys

    sys.path.insert(0, "/project")
    from jspace_study import prepare_holdout

    runs.reload()
    try:
        return prepare_holdout(protocol, f"/runs/{run_id}", runs.commit)
    finally:
        runs.commit()
        cache.commit()


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=131072,
    timeout=3600,
    retries=0,
    single_use_containers=True,
    volumes={"/cache": cache, "/runs": runs},
)
def gpu_worker(protocol, run_id, phase, calibration="", resume=""):
    import sys

    sys.path.insert(0, "/project")
    from jspace_next import calibrate
    from jspace_study import compare_chunk, finish_test, profile

    runs.reload()
    try:
        if phase == "profile":
            return profile(protocol, f"/runs/{run_id}", runs.commit)
        if phase == "calibration":
            return calibrate(protocol, f"/runs/{run_id}", runs.commit, resume)
        if phase == "train":
            return compare_chunk(protocol, calibration, f"/runs/{run_id}", runs.commit, resume)
        if phase == "test":
            return finish_test(protocol, calibration, resume, f"/runs/{run_id}", runs.commit)
        raise ValueError("Unknown study phase")
    finally:
        runs.commit()
        cache.commit()


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=86400,
    retries=0,
    single_use_containers=True,
    volumes={"/runs": runs},
)
def controller(protocol, calibration_protocol, run_id, ledger, handles, continuation=None):
    import sys
    import time

    sys.path.insert(0, "/project")
    from latent_workspace.budget import reserve
    from latent_workspace.train import write_json

    runs.reload()
    root = Path(f"/runs/{run_id}")
    root.mkdir(exist_ok=continuation is not None)
    if continuation is None:
        write_json(root / "budget.json", ledger)
    state = {
        "status": "running",
        "phase": "calibration",
        "children": [],
        "protocol": protocol,
        "calibration_protocol": calibration_protocol,
        "calibration_parent": f"/runs/{handles['calibration']['run_id']}",
        "training_parent": "",
        "calibration_chunks": 1,
        "training_chunks": 0,
        "heldout_call_id": handles["holdout"]["call_id"],
        "profile_call_id": handles["profile"]["call_id"],
    }
    if continuation is not None:
        state = continuation
        state["status"] = "running"
    started = time.monotonic()

    def persist():
        state["elapsed_seconds"] = time.monotonic() - started
        state["reserved_usd"] = sum(
            r["usd"] for r in json.loads((root / "budget.json").read_text())["reservations"]
        )
        write_json(root / "state.json", state)
        runs.commit()

    def wait_call(call_id):
        call = modal.FunctionCall.from_id(call_id)
        while True:
            try:
                return call.get(timeout=30)
            except TimeoutError:
                persist()

    def launch(phase, child_id, calibration="", resume=""):
        reserve(root / "budget.json", child_id, "L40S", timeout_seconds=3600)
        state["phase"] = phase
        state["pending_child"] = {
            "run_id": child_id,
            "phase": phase,
            "calibration": calibration,
            "resume": resume,
        }
        persist()  # Persist the reservation before submission.
        cfg = calibration_protocol if phase == "calibration" else protocol
        call = gpu_worker.spawn(cfg, child_id, phase, calibration, resume)
        state["pending_child"]["call_id"] = call.object_id
        persist()
        result = wait_call(call.object_id)
        state["children"].append({**state.pop("pending_child"), "result": result})
        persist()
        return result

    def handoff():
        generation = state.get("controller_continuations", 0) + 1
        if generation > 3:
            raise TimeoutError("Study exceeded four bounded controller lifetimes")
        reserve(
            root / "budget.json",
            f"{run_id}-continuation-{generation}",
            "CPU",
            timeout_seconds=86400,
        )
        state["controller_continuations"] = generation
        state["status"] = "handoff to next bounded controller"
        persist()
        call = controller.spawn(protocol, calibration_protocol, run_id, None, handles, state)
        write_json(root / f"handoff-{generation}.json", {"call_id": call.object_id})
        runs.commit()
        return {
            "status": "controller handoff",
            "next_controller_call_id": call.object_id,
            "remote_state": str(root / "state.json"),
        }

    try:
        persist()
        profile_result = wait_call(handles["profile"]["call_id"])
        if (
            not profile_result["initial_predictions_identical"]
            or profile_result["future_feature_perturbation_max_nll_difference"] != 0
        ):
            raise ValueError("GPU safeguard check failed")
        state["projection_profile"] = profile_result
        wait_call(handles["holdout"]["call_id"])
        calibration_result = state.get("calibration_result") or wait_call(
            handles["calibration"]["call_id"]
        )
        state["calibration_result"] = calibration_result
        while not calibration_result["completed"]:
            if time.monotonic() - started > 22 * 3600:
                return handoff()
            count = calibration_result["counts"]
            state["calibration_chunks"] += 1
            child_id = f"jspace-calibration-1000-{state['calibration_chunks']:02d}"
            calibration_result = launch("calibration", child_id, resume=state["calibration_parent"])
            if calibration_result["counts"] <= count:
                raise ValueError("Calibration made no checkpoint progress")
            state["calibration_result"] = calibration_result
            state["calibration_parent"] = f"/runs/{child_id}"
            state["calibrated_prompts"] = calibration_result["counts"]
            persist()
        state["calibrated_prompts"] = calibration_result["counts"]
        result = state.get("training_result", {"completed": False})
        while not result["completed"]:
            if time.monotonic() - started > 22 * 3600:
                return handoff()
            state["training_chunks"] += 1
            child_id = f"jspace-matched-30k-{state['training_chunks']:02d}"
            previous_step = state.get("training_step", -1)
            result = launch(
                "train", child_id, state["calibration_parent"], state["training_parent"]
            )
            if result["step"] <= previous_step:
                raise ValueError("Training made no checkpoint progress")
            state["training_result"] = result
            state["training_parent"] = f"/runs/{child_id}"
            state["training_step"] = result["step"]
            persist()
        result = launch(
            "test",
            "jspace-matched-fresh-test-01",
            state["calibration_parent"],
            state["training_parent"],
        )
        state["status"] = "completed"
        state["result"] = result
        persist()
        return state
    except Exception as exc:
        state["status"] = "blocked"
        state["error"] = repr(exc)
        persist()
        raise


@app.local_entrypoint()
def main(protocol: str, run_id: str, phase: str = "profile"):
    cfg = json.loads(Path(protocol).read_text())
    if phase == "controller":
        handles = {
            kind: json.loads(Path(f"artifacts/jspace-{kind}-handle.json").read_text())
            for kind in ("holdout", "profile")
        }
        handles["calibration"] = json.loads(Path("artifacts/jspace-check-handle.json").read_text())
        calibration_cfg = json.loads(Path("configs/jspace-calibration-full.json").read_text())
        reserve("artifacts/budget.json", run_id, "CPU", timeout_seconds=86400)
        ledger = json.loads(Path("artifacts/budget.json").read_text())
        call = controller.spawn(cfg, calibration_cfg, run_id, ledger, handles)
        handle = {
            "run_id": run_id,
            "call_id": call.object_id,
            "remote_budget": f"/runs/{run_id}/budget.json",
        }
        Path("artifacts/budget_remote_controller.json").write_text(
            json.dumps(handle, indent=2) + "\n"
        )
    else:
        reserve(
            "artifacts/budget.json",
            run_id,
            "CPU" if phase == "holdout" else "L40S",
            timeout_seconds=3600,
        )
        call = (
            holdout.spawn(cfg, run_id)
            if phase == "holdout"
            else gpu_worker.spawn(cfg, run_id, phase)
        )
        handle = {"run_id": run_id, "call_id": call.object_id}
    Path(f"artifacts/jspace-{phase}-handle.json").write_text(json.dumps(handle, indent=2) + "\n")
    print(call.object_id, flush=True)
