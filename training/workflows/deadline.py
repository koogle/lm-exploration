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


def tracked_calls(run_folder, kind=None):
    folder = Path(run_folder) / "calls"
    if not folder.exists():
        return []
    records = [json.loads(p.read_text()) for p in folder.glob("*.json")]
    return sorted(r["call_id"] for r in records if kind is None or r["kind"] == kind)


def calls_finished(call_ids):
    """True once every call has returned or failed (none still queued or running)."""
    import modal

    for call_id in call_ids:
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            return False
        except Exception:  # noqa: BLE001, S110 - a failed or cancelled call has finished
            pass
    return True


def cancel_at_deadline(deadline_unix, calls, done=lambda: False, poll_seconds=300, cancel=None):
    """Independent supervisor body: wait for the deadline, then cancel outstanding calls.

    `calls()` returns the call IDs to cancel (read after any volume reload it needs);
    `done()` ends the wait early once the work has finished. Returns the cancelled IDs,
    or None when the work finished first.
    """
    if cancel is None:
        import modal

        def cancel(call_id):
            modal.FunctionCall.from_id(call_id).cancel()

    while (remaining := deadline_unix - time.time()) > 0:
        time.sleep(min(remaining, poll_seconds))
        if done():
            return None
    cancelled = list(calls())
    for call_id in cancelled:
        cancel(call_id)
    return cancelled
