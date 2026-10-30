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


def stop_tracked(run_id, calls=True):
    """Terminate this launch's recorded sandboxes and, with `calls`, cancel its calls."""
    from workflows.deadline import tracked_calls

    stopped = []
    for object_id in tracked_calls(Path("/runs/.calls") / run_id, kind="sandbox"):
        modal.Sandbox.from_id(object_id).terminate()
        stopped.append(object_id)
    if calls:
        for call_id in tracked_calls(Path("/runs/.calls") / run_id):
            if call_id not in stopped:
                modal.FunctionCall.from_id(call_id).cancel()
                stopped.append(call_id)
    return stopped


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
    status = Path("/runs") / run_id / "workflow.json"
    preparation = status.with_name("preparation.json")

    def stage(path):
        return json.loads(path.read_text())["stage"] if path.exists() else None

    while (remaining := deadline_unix - time.time()) > 0:
        time.sleep(min(remaining, 300))
        run_volume.reload()
        # Overlapped preparation can outlast training only if it is still preparing.
        if stage(status) in ("completed", "failed") and stage(preparation) != "preparing":
            return {"stage": stage(status), "terminated": stop_tracked(run_id, calls=False)}
    run_volume.reload()
    cancelled = stop_tracked(run_id)
    record(run_id, "deadline_cancelled", cancelled_calls=cancelled)
    return {"stage": "deadline_cancelled", "cancelled_calls": cancelled}
