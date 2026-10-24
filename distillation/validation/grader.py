"""Grade candidate code against hidden tests, one subprocess per case.

Reads a JSON list of cases ``{"key", "code", "setup", "tests"}`` on stdin and writes
one JSON object mapping each key to ``{"passed", "error"}``. Each case runs in its
own interpreter with a timeout, since generated code can loop. A case passes only if
every test runs and the process then prints its random marker, so code that exits
early cannot count as a pass.
"""

import json
import secrets
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

RUNNER = r"""
import json, sys
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
    with tempfile.TemporaryDirectory() as directory:  # stray files stay out of the cwd
        try:
            done = subprocess.run(
                [sys.executable, "-c", RUNNER],
                check=False,
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=directory,
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


def grade(cases, workers=8, timeout=10):
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda case: run_case(case, timeout), cases))
    return {case["key"]: result for case, result in zip(cases, results)}


if __name__ == "__main__":
    print(json.dumps(grade(json.loads(sys.stdin.read()))))
