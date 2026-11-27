"""Where an interrupted training session continues.

A session writes `step-NNNNNNNNN.pt` checkpoints with `latest.json` while it trains and
`report.json` when it ends. A preempted container leaves checkpoints without a report.
Modal then runs the session's call again with the same input, and a later launch may
start a new attempt; both continue from the newest checkpoint instead of repeating the
session. `run_session` runs a session's processes and treats a session whose report is
written as finished, terminating processes that do not exit within a grace period.
"""

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

SESSION = re.compile(r"(?P<run>.+)-s(?P<index>\d{2})")


def newest_checkpoint(folder):
    """The checkpoint a session folder's `latest.json` names, if it is on disk."""
    latest = Path(folder) / "latest.json"
    if not latest.is_file():
        return None
    checkpoint = Path(folder) / json.loads(latest.read_text())["checkpoint"]
    return checkpoint if checkpoint.is_file() and checkpoint.stat().st_size else None


def session_state(output):
    """How a session call finds its folder: ("fresh", None) when it does not exist,
    ("finished", report) when the session already ended, ("interrupted", checkpoint)
    when an earlier container was cut off (`checkpoint` is None before the first save).
    """
    output = Path(output)
    if not output.exists():
        return "fresh", None
    report = output / "report.json"
    if report.is_file():
        return "finished", json.loads(report.read_text())
    return "interrupted", newest_checkpoint(output)


def continued_checkpoint(runs_root, resume):
    """`resume` (`<run>-sNN/step-NNNNNNNNN.pt`), advanced through every later session of
    the same run that saved a checkpoint. Sessions after the last reported one exist
    only when they were interrupted or stopped, so their newest checkpoint is the
    furthest point of the lineage."""
    match = SESSION.fullmatch(Path(resume).parts[0]) if resume else None
    if match is None:
        return resume
    run, index = match["run"], int(match["index"])
    while True:
        folder = Path(runs_root) / f"{run}-s{index + 1:02d}"
        checkpoint = newest_checkpoint(folder)
        if checkpoint is None:
            return resume
        resume = f"{folder.name}/{checkpoint.name}"
        index += 1


# Seconds a session's processes get to exit after its report is written.
EXIT_GRACE_SECONDS = 600


def run_session(command, log, report, grace=EXIT_GRACE_SECONDS, poll=10, **popen):
    """Run a training session's `command` (output to the open file `log`) and return its
    exit code. Once a `report` file newer than the launch exists, the session has ended
    and saved its checkpoint: processes still running `grace` seconds later (a rank
    blocked in shutdown) are terminated, and the session counts as finished (code 0)."""
    started = time.time()
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, **popen
    )
    reported = None
    while True:
        try:
            return process.wait(timeout=poll)
        except subprocess.TimeoutExpired:
            pass
        report_path = Path(report)
        if reported is None and report_path.is_file() and report_path.stat().st_mtime >= started:
            reported = time.time()
        if reported is not None and time.time() - reported >= grace:
            log.write(
                f"Session reported but its processes were still running {grace} s later; "
                "terminating them.\n"
            )
            log.flush()
            for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
                os.killpg(process.pid, sig)
                try:
                    process.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
            return 0
