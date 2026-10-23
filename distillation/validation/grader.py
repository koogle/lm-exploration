"""Execute candidate code against hidden tests. Runs inside a network-blocked sandbox.

Reads a JSON list of cases ``{"key", "code", "setup", "tests"}`` on stdin and writes
one JSON object mapping each key to ``{"passed", "error"}``. Each case runs in its
own isolated interpreter with CPU, memory, file-size and process limits, an empty
environment and a fresh temporary directory. A case passes only if every test
passes and the process prints its random marker.
"""

import json
import secrets
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

RUNNER = r"""
import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
case = json.loads(sys.stdin.read())
namespace = {"__name__": "candidate"}
exec(compile(case["setup"] + "\n" + case["code"], "candidate", "exec"), namespace)
for test in case["tests"]:
    exec(compile(test, "test", "exec"), namespace)
sys.stdout.write(case["marker"])
"""


def run_case(case, timeout=10):
    marker = secrets.token_hex(16)
    payload = json.dumps({**case, "marker": marker})
    with tempfile.TemporaryDirectory() as directory:
        try:
            done = subprocess.run(
                [sys.executable, "-I", "-S", "-c", RUNNER],
                check=False,
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=directory,
                env={},
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "timeout"}
    if done.returncode == 0 and done.stdout.endswith(marker):
        return {"passed": True, "error": None}
    lines = done.stderr.strip().splitlines()
    return {
        "passed": False,
        "error": (lines[-1] if lines else f"exit {done.returncode}")[:200],
    }


def grade(cases, workers=8):
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(run_case, cases))
    return {case["key"]: result for case, result in zip(cases, results)}


if __name__ == "__main__":
    print(json.dumps(grade(json.loads(sys.stdin.read()))))
