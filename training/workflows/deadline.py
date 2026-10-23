"""One 24-hour deadline per workflow launch, shared by every call it starts.

The launcher fixes `deadline_unix` before anything runs. Each step refuses new work
once too little time remains and bounds child timeouts by what is left. Every
spawned call is recorded under the run folder so the independent supervisor can
cancel outstanding work at the deadline.
"""

import json
import time
from pathlib import Path

MAX_LAUNCH_SECONDS = 24 * 60 * 60
# The supervisor cancels this long before 24 hours, inside its own 24-hour timeout.
SUPERVISOR_MARGIN_SECONDS = 900
# Time a training segment needs to stop, checkpoint and exit before its timeout.
SHUTDOWN_SECONDS = 480
MIN_WORK_SECONDS = 900


def new_deadline(now=None):
    return (time.time() if now is None else now) + MAX_LAUNCH_SECONDS - SUPERVISOR_MARGIN_SECONDS


def bounded_timeout(deadline_unix, requested, now=None):
    """Return a child timeout that ends before the launch deadline, or refuse the work."""
    remaining = int(deadline_unix - (time.time() if now is None else now))
    if remaining < MIN_WORK_SECONDS:
        raise RuntimeError("Launch deadline reached; refusing new work")
    return min(requested, remaining)


def segment_seconds(timeout, requested):
    """Training time inside a bounded segment timeout, leaving room to checkpoint."""
    seconds = min(requested, timeout - SHUTDOWN_SECONDS)
    if seconds < 1:
        raise RuntimeError("Segment timeout leaves no training time")
    return seconds


def track(run_folder, call_id, kind):
    """Record one spawned call; one file per call avoids read-modify-write races."""
    folder = Path(run_folder) / "calls"
    folder.mkdir(parents=True, exist_ok=True)
    record = {"call_id": call_id, "kind": kind, "spawned_unix": time.time()}
    (folder / f"{call_id}.json").write_text(json.dumps(record) + "\n")


def tracked_calls(run_folder):
    folder = Path(run_folder) / "calls"
    if not folder.exists():
        return []
    return sorted(json.loads(p.read_text())["call_id"] for p in folder.glob("*.json"))
