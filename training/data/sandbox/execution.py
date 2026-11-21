"""Check model-written Python functions against unit tests, only inside a Modal Sandbox.

`program` joins a solution and its tests into one script, in one of three test styles:

- `functions` (the default): the solution becomes the module `solution` and its names
  are visible to the tests, the tests run, every `test*` function they define (not ones
  imported from the solution) is called, and the script prints a nonce last.
- `pytest`: the solution is written as `solution.py` beside the tests as
  `test_solution.py` in the program's own folder, pytest runs them (fixtures, parametrization,
  `pytest.raises`), and the script prints the nonce last only if every test passed.
- `contest`: input/output cases graded as LiveCodeBench grades them
  (`evaluation.benchmarks.livecodebench.grade`). The tests are the JSON object
  `{"functional": bool, "method": name, "cases": [[input, expected output], ...]}`
  (`contest_tests`). Each case runs in its own process, as LiveCodeBench's runner does:
  a stdin program as `__main__` with the case's input as its standard input, its output
  matched by whitespace-separated tokens; a functional solution's `Solution().<method>`
  called with the case's JSON lines as arguments, its JSON result equal to the expected
  one. The script prints the nonce only if every case matched; it has `CONTEST_SECONDS`
  for all cases and each case `CONTEST_CASE_SECONDS`.
 `runner.py` runs the
scripts with resource limits and passes a script only if it exits cleanly and prints its
nonce, so a solution that exits early, raises or skips the tests fails.

`Sandbox` runs batches in one network-blocked sandbox with no volumes or secrets, whose
lifetime is capped by its `timeout` (at most the caller's remaining launch budget).
`run_local` runs a batch with the same runner in a local subprocess, for tests of our own
fixtures only.
"""

import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).with_name("runner.py")
SANDBOX_RUNNER = "/runner/runner.py"
# Seconds each program may run.
PROGRAM_SECONDS = 5
# Seconds a `contest` program has for all of its cases.
CONTEST_SECONDS = 20
TEST_STYLES = ("functions", "pytest", "contest")
# Characters of a request written to the sandbox's standard input at a time.
STDIN_CHUNK = 1 << 20
# What LiveCodeBench's runner gives functional solutions.
CONTEST_PRELUDE = (
    "from typing import *\nfrom collections import *\n"
    "import math, heapq, bisect, itertools, functools\n"
)
CONTEST_PROGRAM = """import json, subprocess, sys
_tests = json.loads({tests!r})
with open("solution.py", "w") as _file:
    _file.write({solution!r})
with open("case.py", "w") as _file:
    _file.write({case!r})
def _same(expected, actual):
    if _tests["functional"]:
        try:
            return json.loads(expected) == json.loads(actual)
        except json.JSONDecodeError:
            return expected.strip() == actual.strip()
    return expected.split() == actual.split()
_mode = "functional" if _tests["functional"] else "stdin"
for _input, _expected in _tests["cases"]:
    try:
        _result = subprocess.run(
            [sys.executable, "-I", "case.py", _mode, _tests["method"]],
            input=_input, capture_output=True, text=True, timeout={case_seconds},
        )
    except subprocess.TimeoutExpired:
        sys.exit(1)
    if _result.returncode != 0 or not _same(_expected, _result.stdout):
        sys.exit(1)
print({nonce!r})
"""
# One case, in its own process: LiveCodeBench's runner without its resource limits (the
# sandbox runner's limits are inherited).
CONTEST_CASE = """import json, sys
sys.setrecursionlimit(100000)
mode, method = sys.argv[1], sys.argv[2]
source = open("solution.py").read()
if mode == "stdin":
    exec(compile(source, "solution.py", "exec"), {"__name__": "__main__"})
else:
    scope = {"__name__": "solution"}
    exec(compile(PRELUDE + source, "solution.py", "exec"), scope)
    args = [json.loads(line) for line in sys.stdin.read().splitlines() if line.strip()]
    print(json.dumps(getattr(scope["Solution"](), method)(*args)))
""".replace("PRELUDE", repr(CONTEST_PRELUDE))
# Seconds per case, as LiveCodeBench's grader allows.
CONTEST_CASE_SECONDS = 8


def contest_tests(cases, starter_code):
    """The `contest` tests of a problem with (input, expected output) `cases`: functional
    when it has LeetCode starter code, calling its last `self` method."""
    names = re.findall(r"def\s+(\w+)\s*\(\s*self", starter_code or "")
    functional = bool(starter_code)
    return json.dumps(
        {"functional": functional, "method": names[-1] if names else "", "cases": cases}
    )


def program(solution, tests, nonce, style="functions"):
    """The script that runs `tests` against `solution` in `style` and prints `nonce`."""
    if style not in TEST_STYLES:
        raise ValueError(f"test style must be one of {TEST_STYLES}")
    if style == "contest":
        return CONTEST_PROGRAM.format(
            tests=tests,
            solution=solution,
            case=CONTEST_CASE,
            case_seconds=CONTEST_CASE_SECONDS,
            nonce=nonce,
        )
    if style == "pytest":
        return (
            "import os, sys\n"
            "import pytest\n"
            "_folder = os.getcwd()\n"
            "for _name, _text in (('solution.py', "
            f"{solution!r}), ('test_solution.py', {tests!r})):\n"
            "    with open(os.path.join(_folder, _name), 'w') as _file:\n"
            "        _file.write(_text)\n"
            "sys.path.insert(0, _folder)\n"
            "_args = [os.path.join(_folder, 'test_solution.py'), '-q', '-p', 'no:cacheprovider']\n"
            "if pytest.main(_args) == 0:\n"
            f"    print({nonce!r})\n"
        )
    return (
        "import sys, types\n"
        "_module = types.ModuleType('solution')\n"
        f"exec(compile({solution!r}, 'solution.py', 'exec'), _module.__dict__)\n"
        "sys.modules['solution'] = _module\n"
        "globals().update({k: v for k, v in vars(_module).items() if not k.startswith('__')})\n"
        f"exec(compile({tests!r}, 'tests.py', 'exec'), globals())\n"
        "for _name, _test in list(globals().items()):\n"
        "    if _name.startswith('test') and getattr(_test, '__module__', '') == '__main__':\n"
        "        _test()\n"
        f"print({nonce!r})\n"
    )


def _request(pairs, seconds, python=None):
    """`pairs`: (solution, tests) or (solution, tests, test style)."""
    programs = []
    for solution, tests, *style in pairs:
        nonce = secrets.token_hex(16)
        entry = {"source": program(solution, tests, nonce, *style), "nonce": nonce}
        if style == ["contest"]:
            entry["timeout"] = max(seconds, CONTEST_SECONDS)
        programs.append(entry)
    return json.dumps(
        {"programs": programs, "timeout": seconds, **({"python": python} if python else {})}
    )


def run_local(pairs, seconds=PROGRAM_SECONDS):
    """Results for (solution, tests) pairs, run by `runner.py` in a subprocess. As
    root the runner gives programs unprivileged user IDs, which may not reach this
    environment's interpreter, so programs then run with the system `/usr/bin/python3`."""
    python = "/usr/bin/python3" if os.geteuid() == 0 else None
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        input=_request(pairs, seconds, python),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def image():
    import modal

    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy==2.2.6", "pytest==8.4.2")
        .add_local_file(RUNNER, SANDBOX_RUNNER)
    )


class Sandbox:
    """One code sandbox; `close()` terminates it. Its ID is `object_id`."""

    def __init__(self, app, seconds, cpu=16):
        import modal

        self.sandbox = modal.Sandbox.create(
            app=app,
            image=image(),
            timeout=int(seconds),
            cpu=cpu,
            memory=32768,
            block_network=True,
        )
        self.object_id = self.sandbox.object_id

    def run(self, pairs, seconds=PROGRAM_SECONDS):
        """Results for (solution, tests) pairs, in order."""
        if not pairs:
            return []
        process = self.sandbox.exec("python", SANDBOX_RUNNER)
        request = _request(pairs, seconds)
        # The stream buffers a bounded amount: write the request in drained chunks.
        for start in range(0, len(request), STDIN_CHUNK):
            process.stdin.write(request[start : start + STDIN_CHUNK])
            process.stdin.drain()
        process.stdin.write_eof()
        process.stdin.drain()
        output = process.stdout.read()
        if process.wait() != 0:
            raise RuntimeError(f"Code runner failed: {process.stderr.read()[-2000:]}")
        return json.loads(output)

    def close(self):
        self.sandbox.terminate()
