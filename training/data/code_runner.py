"""Run untrusted Python programs and report which finished cleanly. Standard library only.

This file is copied into the code sandbox (`training_data.code_execution`) and run there
as `python code_runner.py`: it reads `{"programs": [{"source", "nonce"}], "timeout"}`
(and optionally `"python"`, the interpreter for programs; default this one) as JSON on stdin and writes one result per program as JSON on stdout. Each program runs as
`python -I -` (source on stdin, so it has no `__file__`) in its own empty directory with
CPU, memory and file-size limits, in its own process group, which is killed when it ends.
When the runner is root (as in the sandbox), each concurrently running program gets its
own unprivileged user ID and a private directory, so it cannot read, change or delete
another program's files or the runner's output capture. A program passes only if it
exits with code 0 and the last line it prints is its nonce; a program the runner could
not run at all fails with a `runner error` status instead of stopping the batch.
"""

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

MEMORY_BYTES = 1024**3
FILE_BYTES = 16 * 1024**2
OUTPUT_TAIL = 2048
# First of the unprivileged user IDs given to concurrently running programs.
FIRST_UID = 20000


def prelude(seconds):
    """Hard CPU, memory and file-size limits, set before the program's own code runs; an
    unprivileged process cannot raise a hard limit again."""
    return (
        "import resource as _resource\n"
        f"_resource.setrlimit(_resource.RLIMIT_CPU, ({seconds + 1}, {seconds + 1}))\n"
        f"_resource.setrlimit(_resource.RLIMIT_AS, ({MEMORY_BYTES}, {MEMORY_BYTES}))\n"
        f"_resource.setrlimit(_resource.RLIMIT_FSIZE, ({FILE_BYTES}, {FILE_BYTES}))\n"
        "del _resource\n"
    )


def run(program, seconds, uid, python):
    folder = tempfile.mkdtemp()
    process = None
    try:
        if uid is not None:
            os.chown(folder, uid, uid)
        # An unlinked file: the program cannot remove or replace the captured output.
        with tempfile.TemporaryFile() as stdout:
            process = subprocess.Popen(
                [python, "-I", "-"],
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=subprocess.PIPE,
                cwd=folder,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                process_group=0,
                **({"user": uid, "group": uid, "extra_groups": []} if uid is not None else {}),
            )
            try:
                source = prelude(seconds) + program["source"]
                _, stderr = process.communicate(source.encode(), timeout=seconds)
            except subprocess.TimeoutExpired:
                return {"passed": False, "status": "timeout", "stderr": ""}
            stdout.seek(max(0, stdout.seek(0, 2) - OUTPUT_TAIL))
            tail = stdout.read().decode(errors="replace")
    except Exception as error:  # noqa: BLE001 - one broken program must not end the batch
        return {"passed": False, "status": f"runner error: {error!r}"[:200], "stderr": ""}
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        shutil.rmtree(folder, ignore_errors=True)
    lines = tail.strip().splitlines()
    passed = process.returncode == 0 and bool(lines) and lines[-1].strip() == program["nonce"]
    return {
        "passed": passed,
        "status": "pass" if passed else f"exit {process.returncode}",
        "stderr": stderr[-OUTPUT_TAIL:].decode(errors="replace"),
    }


def run_all(request):
    seconds, python = request["timeout"], request.get("python", sys.executable)
    workers = os.cpu_count() or 1
    # One user ID per concurrent slot, when running as root.
    slots = queue.Queue()
    for slot in range(workers):
        slots.put(FIRST_UID + slot if os.geteuid() == 0 else None)

    def one(program):
        uid = slots.get()
        try:
            return run(program, seconds, uid, python)
        finally:
            slots.put(uid)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, request["programs"]))


if __name__ == "__main__":
    json.dump(run_all(json.load(sys.stdin)), sys.stdout)
