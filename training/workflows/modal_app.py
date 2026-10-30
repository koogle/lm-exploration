"""The Modal app every workflow stage runs in: volumes, run records, call tracking and
the deadline supervisor. Importing this module never starts a job."""

import json
import time
from pathlib import Path

import modal

app = modal.App("lm-exploration-training-workflows")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
run_volume = modal.Volume.from_name("lm-exploration-training")
supervisor_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONPATH": "/project"})
    .add_local_dir("workflows", "/project/workflows", ignore=["__pycache__"])
)


def track_call(run_id, call_id, kind):
    """Record a call (or, with kind `"sandbox"`, a sandbox) so the deadline supervisor can
    stop it."""
    from workflows.deadline import track

    track(Path("/runs/.calls") / run_id, call_id, kind)
    run_volume.commit()


def record(run_id, stage, *, status_file="workflow.json", **details):
    folder = Path("/runs") / run_id
    folder.mkdir(parents=True, exist_ok=True)
    status = {"run_id": run_id, "stage": stage, "updated_unix": time.time(), **details}
    (folder / status_file).write_text(json.dumps(status, indent=2) + "\n")
    run_volume.commit()
    print(json.dumps(status), flush=True)


def terminate_sandboxes(run_id):
    """Terminate this launch's recorded sandboxes; returns their IDs."""
    from workflows.deadline import tracked_calls

    sandboxes = tracked_calls(Path("/runs/.calls") / run_id, kind="sandbox")
    for object_id in sandboxes:
        modal.Sandbox.from_id(object_id).terminate()
    return sandboxes


@app.function(
    image=supervisor_image,
    cpu=0.25,
    memory=512,
    timeout=86400,
    retries=0,
    volumes={"/runs": run_volume},
)
def enforce_deadline(run_id, deadline_unix):
    """Independent supervisor: cancel every recorded call and terminate every recorded
    sandbox of this launch at its deadline. A launch that records `completed` or `failed`
    first only has its sandboxes terminated."""
    from workflows.deadline import cancel_at_deadline, tracked_calls

    calls_folder = Path("/runs/.calls") / run_id
    status = Path("/runs") / run_id / "workflow.json"
    preparation = status.with_name("preparation.json")

    def stage(path):
        return json.loads(path.read_text())["stage"] if path.exists() else None

    def done():
        run_volume.reload()
        # Overlapped preparation can outlast training only if it is still preparing.
        return stage(status) in ("completed", "failed") and stage(preparation) != "preparing"

    def calls():
        run_volume.reload()
        return tracked_calls(calls_folder)

    def cancel(object_id):
        if object_id in tracked_calls(calls_folder, kind="sandbox"):
            modal.Sandbox.from_id(object_id).terminate()
        else:
            modal.FunctionCall.from_id(object_id).cancel()

    cancelled = cancel_at_deadline(deadline_unix, calls, done, cancel=cancel)
    if cancelled is None:
        return {"stage": stage(status), "terminated": terminate_sandboxes(run_id)}
    record(run_id, "deadline_cancelled", cancelled_calls=cancelled)
    return {"stage": "deadline_cancelled", "cancelled_calls": cancelled}
