"""Check model-written Python functions against unit tests, only inside a Modal Sandbox.

`program` joins a solution and its tests into one script: the solution becomes the
module `solution` and its names are visible to the tests, the tests run, every `test*` function they define (not ones imported
from the solution) is called, and the script prints a nonce last. `code_runner.py` runs the scripts with resource limits
and passes a script only if it exits cleanly and prints its nonce, so a solution that
exits early, raises or skips the tests fails.

`Sandbox` runs batches in one network-blocked sandbox with no volumes or secrets, whose
lifetime is capped by its `timeout` (at most the caller's remaining launch budget).
`run_local` runs a batch with the same runner in a local subprocess, for tests of our own
fixtures only.
"""

import json
import secrets
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).with_name("code_runner.py")
SANDBOX_RUNNER = "/runner/code_runner.py"
# Seconds each program may run.
PROGRAM_SECONDS = 5


def program(solution, tests, nonce):
    """The script that imports `solution` as a module, runs `tests` and prints `nonce`."""
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


def _request(pairs, seconds):
    programs = []
    for solution, tests in pairs:
        nonce = secrets.token_hex(16)
        programs.append({"source": program(solution, tests, nonce), "nonce": nonce})
    return json.dumps({"programs": programs, "timeout": seconds})


def run_local(pairs, seconds=PROGRAM_SECONDS):
    """Results for (solution, tests) pairs, run by `code_runner.py` in a subprocess."""
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        input=_request(pairs, seconds),
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
        process.stdin.write(_request(pairs, seconds))
        process.stdin.write_eof()
        process.stdin.drain()
        output = process.stdout.read()
        if process.wait() != 0:
            raise RuntimeError(f"Code runner failed: {process.stderr.read()[-2000:]}")
        return json.loads(output)

    def close(self):
        self.sandbox.terminate()
