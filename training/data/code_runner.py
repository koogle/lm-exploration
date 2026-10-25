"""Run untrusted Python programs and report which finished cleanly. Standard library only.

This file is copied into the code sandbox (`training_data.code_execution`) and run there
as `python code_runner.py`: it reads `{"programs": [{"source", "nonce"}], "timeout"}` as
JSON on stdin and writes one result per program as JSON on stdout. Each program runs as
`python -I -` (source on stdin, so it has no `__file__`) in its own empty directory with
CPU, memory, file-size and process limits. A program passes only if it exits with code
0 and the last line it prints is its nonce.
"""

import json
import os
import resource
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

MEMORY_BYTES = 1024**3
FILE_BYTES = 16 * 1024**2
OUTPUT_TAIL = 2048


def limits(seconds):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (seconds + 1, seconds + 1))
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
        os.setsid()

    return apply


def run(program, seconds):
    with tempfile.TemporaryDirectory() as folder:
        stdout_path = os.path.join(folder, ".stdout")
        with open(stdout_path, "w+b") as stdout:
            try:
                process = subprocess.run(
                    [sys.executable, "-I", "-"],
                    check=False,
                    input=program["source"].encode(),
                    stdout=stdout,
                    stderr=subprocess.PIPE,
                    cwd=folder,
                    timeout=seconds,
                    preexec_fn=limits(seconds),
                    env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                )
            except subprocess.TimeoutExpired:
                return {"passed": False, "status": "timeout", "stderr": ""}
            stdout.seek(max(0, stdout.seek(0, 2) - OUTPUT_TAIL))
            tail = stdout.read().decode(errors="replace")
    lines = tail.strip().splitlines()
    passed = process.returncode == 0 and bool(lines) and lines[-1].strip() == program["nonce"]
    return {
        "passed": passed,
        "status": "pass" if passed else f"exit {process.returncode}",
        "stderr": process.stderr[-OUTPUT_TAIL:].decode(errors="replace"),
    }


def run_all(request):
    seconds = request["timeout"]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        return list(pool.map(lambda p: run(p, seconds), request["programs"]))


if __name__ == "__main__":
    json.dump(run_all(json.load(sys.stdin)), sys.stdout)
