"""Runs inside a disposable PR-task sandbox; standard library only. Not imported elsewhere.

    python3 pr_task_runner.py setup <task.json>   # network on: check out, install
    python3 pr_task_runner.py test <task.json>    # network off: run tests twice

`setup` clones the bare mirror into /work/repo at the commit and installs the
project's dependencies with heuristics per language (uv for Python, the lockfile's
package manager for JavaScript/TypeScript, module download for Go). `test` runs
the commit's test files with source files reverted to the parent ("before") and at
the commit ("after"), and prints one JSON line: per-test outcomes for both states
plus the commands and log tails. The caller decides which tests are fail-to-pass.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/work/repo")
INSTALL_SECONDS = 900
TEST_SECONDS = 600
TAIL = 4000


def run(command, cwd=REPO, seconds=TEST_SECONDS, env=None):
    try:
        done = subprocess.run(
            command,
            cwd=cwd,
            shell=isinstance(command, str),
            capture_output=True,
            text=True,
            check=False,
            timeout=seconds,
            env={**os.environ, "CI": "1", **(env or {})},
        )
        return done.returncode, done.stdout + done.stderr
    except subprocess.TimeoutExpired as error:
        out = (error.stdout or b"") + (error.stderr or b"")
        return "timeout", out.decode(errors="replace") if isinstance(out, bytes) else (out or "")


# Setup.
def _python_setup(log):
    log.append(
        run("uv venv --python-preference only-managed -q .venv || uv venv -q .venv", seconds=300)
    )
    pip = "uv pip install -q --python .venv/bin/python"
    if (REPO / "uv.lock").exists():
        log.append(
            run(
                "uv sync -q --all-extras --all-groups --frozen",
                seconds=INSTALL_SECONDS,
                env={"UV_PROJECT_ENVIRONMENT": ".venv"},
            )
        )
    elif (REPO / "pyproject.toml").exists() or (REPO / "setup.py").exists():
        for extras in ("[test,tests,testing,dev]", "[test]", "[dev]", ""):
            code, out = run(f"{pip} -e '.{extras}'", seconds=INSTALL_SECONDS)
            log.append((code, out))
            if code == 0:
                break
    for name in (
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "requirements_dev.txt",
        "test-requirements.txt",
    ):
        if (REPO / name).exists():
            log.append(run(f"{pip} -r {name}", seconds=INSTALL_SECONDS))
    log.append(run(f"{pip} pytest", seconds=300))


def _javascript_setup(log):
    if (REPO / "pnpm-lock.yaml").exists():
        command = "corepack pnpm install --frozen-lockfile || corepack pnpm install"
    elif (REPO / "yarn.lock").exists():
        command = "corepack yarn install --immutable || corepack yarn install"
    elif (REPO / "bun.lock").exists() or (REPO / "bun.lockb").exists():
        command = "bun install --frozen-lockfile || bun install"
    elif (REPO / "package-lock.json").exists():
        command = "npm ci --no-audit --no-fund || npm install --no-audit --no-fund"
    else:
        command = "npm install --no-audit --no-fund"
    log.append(run(command, seconds=INSTALL_SECONDS, env={"COREPACK_ENABLE_DOWNLOAD_PROMPT": "0"}))


def _go_setup(log):
    log.append(
        run(
            "go mod download && go build ./... && go vet ./... >/dev/null 2>&1; true",
            seconds=INSTALL_SECONDS,
        )
    )


def setup(task):
    shutil.rmtree(REPO, ignore_errors=True)
    mirror = Path("/work/mirror.git")
    log = [
        run(
            f"git clone -q {mirror} {REPO} && git -C {REPO} checkout -q {task['sha']}",
            cwd="/work",
            seconds=300,
        )
    ]
    if log[0][0] != 0:
        return {"stage": "checkout", "log": [{"code": c, "tail": t[-1500:]} for c, t in log]}
    {"python": _python_setup, "javascript": _javascript_setup, "go": _go_setup}[task["language"]](
        log
    )
    return {"stage": "installed", "log": [{"code": c, "tail": t[-1500:]} for c, t in log]}


# Tests.
def _nearest_package(path):
    folder = (REPO / path).parent
    while folder != REPO.parent:
        if (folder / "package.json").exists():
            return folder
        folder = folder.parent
    return REPO


def _javascript_runner(package):
    manifest = json.loads((package / "package.json").read_text())
    root = (
        json.loads((REPO / "package.json").read_text()) if (REPO / "package.json").exists() else {}
    )
    names = set()
    for data in (manifest, root):
        for key in ("dependencies", "devDependencies"):
            names |= set(data.get(key, {}))
        names |= {w for s in data.get("scripts", {}).values() for w in re.findall(r"[\w-]+", s)}
    for runner in ("vitest", "jest", "mocha"):
        if runner in names:
            return runner
    return "node"


def _python_tests(paths):
    code, out = run(
        [".venv/bin/python", "-m", "pytest", "-rA", "-p", "no:cacheprovider", "-q", *paths]
    )
    results = {}
    for status, test in re.findall(r"^(PASSED|FAILED|ERROR) (\S+)", out, re.MULTILINE):
        results[test] = "pass" if status == "PASSED" else "fail"
    return results, code, out


def _javascript_tests(paths):
    results, codes, outs = {}, [], []
    groups = {}
    for path in paths:
        groups.setdefault(_nearest_package(path), []).append(path)
    for package, files in groups.items():
        relative = [str((REPO / f).relative_to(package)) for f in files]
        runner = _javascript_runner(package)
        report = Path("/tmp/report.json")
        report.unlink(missing_ok=True)
        if runner == "vitest":
            command = [
                "npx",
                "--no-install",
                "vitest",
                "run",
                "--reporter=json",
                f"--outputFile={report}",
                *relative,
            ]
        elif runner == "jest":
            command = [
                "npx",
                "--no-install",
                "jest",
                "--json",
                f"--outputFile={report}",
                "--runTestsByPath",
                *relative,
            ]
        elif runner == "mocha":
            command = [
                "npx",
                "--no-install",
                "mocha",
                "--reporter",
                "json",
                "--reporter-option",
                f"output={report}",
                *relative,
            ]
        else:
            command = ["node", "--test", "--test-reporter=tap", *relative]
        code, out = run(command, cwd=package)
        codes.append(code)
        outs.append(out)
        if report.exists():
            try:
                data = json.loads(report.read_text())
            except json.JSONDecodeError:
                data = {}
            for suite in data.get("testResults", []):
                for test in suite.get("assertionResults", []):
                    name = f"{Path(suite['name']).name}::{test['fullName']}"
                    results[name] = (
                        "pass"
                        if test["status"] == "passed"
                        else (
                            "skip" if test["status"] in ("pending", "skipped", "todo") else "fail"
                        )
                    )
            for key, status in (("passes", "pass"), ("failures", "fail")):
                for test in data.get(key, []):
                    results[f"{Path(test.get('file', '')).name}::{test['fullTitle']}"] = status
        elif runner == "node":
            for status, name in re.findall(r"^\s*(ok|not ok) \d+ - (.+)$", out, re.MULTILINE):
                results[f"{relative[0]}::{name}"] = "pass" if status == "ok" else "fail"
        if not results and code not in (0, "timeout"):
            results.update({f"{f}::<file>": "fail" for f in files})
    return results, codes, "\n".join(outs)


def _go_tests(paths):
    packages = sorted({"./" + str(Path(p).parent) for p in paths})
    code, out = run(["go", "test", "-json", "-count=1", *packages], seconds=TEST_SECONDS)
    results = {}
    for line in out.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("Test") and event.get("Action") in ("pass", "fail", "skip"):
            results[f"{event['Package']}::{event['Test']}"] = event["Action"]
    if not results and code != 0:
        results = {f"{p}::<package>": "fail" for p in packages}
    return results, code, out


TESTERS = {"python": _python_tests, "javascript": _javascript_tests, "go": _go_tests}


def _state(task, before):
    run(["git", "checkout", "-q", "-f", task["sha"], "--", "."])
    if before:
        for path in task["source_paths"]:
            code, _ = run(["git", "checkout", "-q", task["parent"], "--", path])
            if code != 0:  # added by the commit
                (REPO / path).unlink(missing_ok=True)


def test(task):
    out = {}
    for name, before in (("before", True), ("after", False)):
        _state(task, before)
        results, code, log = TESTERS[task["language"]](task["test_paths"])
        out[name] = {"results": results, "code": code, "tail": log[-TAIL:]}
    _state(task, False)
    return out


if __name__ == "__main__":
    stage, path = sys.argv[1], sys.argv[2]
    task = json.loads(Path(path).read_text())
    print("PR_TASK_RESULT " + json.dumps(setup(task) if stage == "setup" else test(task)))
