"""Runs inside a disposable PR-task sandbox; standard library only. Not imported elsewhere.

    python3 pr_task_runner.py setup <task.json>      # network on: check out, install
    python3 pr_task_runner.py test <task.json>       # network off: run tests twice
    python3 pr_task_runner.py workspace <task.json>  # network off: the agent's start
    python3 pr_task_runner.py tool -                 # one agent tool call (JSON on stdin)
    python3 pr_task_runner.py patch -                # the agent's changes as a diff
    python3 pr_task_runner.py grade <task.json>      # apply /work/patch.diff, run tests

`setup` clones the bare mirror into /work/repo at the commit and installs the
project's dependencies with heuristics per language (uv for Python, the lockfile's
package manager for JavaScript/TypeScript, module download for Go). `test` runs
the commit's test files with source files reverted to the parent ("before") and at
the commit ("after"), and prints one JSON line: per-test outcomes for both states
plus the commands and log tails. The caller decides which tests are fail-to-pass.

`workspace` turns an installed sandbox into the agent's starting point: source and
test files at the parent, the mirror, the task file and the git history removed,
and a fresh one-commit repository so `patch` can diff the agent's work. `grade`
runs in a separate sandbox from the installed snapshot: tests at the commit,
sources at the parent plus the agent's patch (its test-file changes dropped).
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


def run(command, cwd=None, seconds=TEST_SECONDS, env=None):
    try:
        done = subprocess.run(
            command,
            cwd=cwd or REPO,
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
def _nearest(path, marker):
    """The closest folder above `path` (inside the repository) holding `marker`, or None."""
    folder = (REPO / path).parent
    while folder != REPO.parent:
        if (folder / marker).exists():
            return folder
        folder = folder.parent
    return None


VENV_PYTHON = REPO / ".venv/bin/python"


def _python_project(paths):
    """The folder of the project that owns the tests (nearest pyproject.toml or setup.py)."""
    for path in paths:
        for marker in ("pyproject.toml", "setup.py"):
            if folder := _nearest(path, marker):
                return folder
    return REPO


def _python_setup(task, log):
    project = _python_project(task["test_paths"])
    log.append(
        run("uv venv --python-preference only-managed -q .venv || uv venv -q .venv", seconds=300)
    )
    pip = f"uv pip install -q --python {VENV_PYTHON}"
    if (project / "uv.lock").exists():
        log.append(
            run(
                "uv sync -q --all-extras --all-groups --frozen",
                cwd=project,
                seconds=INSTALL_SECONDS,
                env={"UV_PROJECT_ENVIRONMENT": str(REPO / ".venv")},
            )
        )
    elif (project / "pyproject.toml").exists() or (project / "setup.py").exists():
        for extras in ("[test,tests,testing,dev]", "[test]", "[dev]", ""):
            code, out = run(f"{pip} -e '.{extras}'", cwd=project, seconds=INSTALL_SECONDS)
            log.append((code, out))
            if code == 0:
                break
    for folder in sorted({project, REPO}):
        for name in (
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt",
            "requirements_dev.txt",
            "test-requirements.txt",
        ):
            if (folder / name).exists():
                log.append(run(f"{pip} -r {name}", cwd=folder, seconds=INSTALL_SECONDS))
    log.append(run(f"{pip} pytest", seconds=300))


LOCKFILES = (
    ("pnpm-lock.yaml", "corepack pnpm install --frozen-lockfile || corepack pnpm install"),
    ("yarn.lock", "corepack yarn install --immutable || corepack yarn install"),
    ("bun.lock", "bun install --frozen-lockfile || bun install"),
    ("bun.lockb", "bun install --frozen-lockfile || bun install"),
    ("package-lock.json", "npm ci --no-audit --no-fund || npm install --no-audit --no-fund"),
)


def _install_root(package):
    """The outermost folder from `package` up to the repository holding a lockfile."""
    root, folder = package, package
    while folder != REPO.parent:
        if any((folder / name).exists() for name, _ in LOCKFILES):
            root = folder
        folder = folder.parent
    return root


def _javascript_setup(task, log):
    packages = {_nearest(p, "package.json") for p in task["test_paths"]} - {None}
    for root in sorted({_install_root(p) for p in packages}):
        command = next(
            (c for name, c in LOCKFILES if (root / name).exists()),
            "npm install --no-audit --no-fund",
        )
        log.append(
            run(
                command,
                cwd=root,
                seconds=INSTALL_SECONDS,
                env={"COREPACK_ENABLE_DOWNLOAD_PROMPT": "0"},
            )
        )


def _go_setup(task, log):
    for module in sorted({_nearest(p, "go.mod") for p in task["test_paths"]} - {None}):
        log.append(run("go mod download && go build ./...", cwd=module, seconds=INSTALL_SECONDS))


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
        task, log
    )
    return {"stage": "installed", "log": [{"code": c, "tail": t[-1500:]} for c, t in log]}


# Tests.
def _javascript_runner(package):
    manifest = (
        json.loads((package / "package.json").read_text())
        if (package / "package.json").exists()
        else {}
    )
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
    project = _python_project(paths)
    relative = [str((REPO / p).relative_to(project)) for p in paths]
    code, out = run(
        [str(VENV_PYTHON), "-m", "pytest", "-rA", "-p", "no:cacheprovider", "-q", *relative],
        cwd=project,
    )
    results = {}
    for status, test in re.findall(r"^(PASSED|FAILED|ERROR) (\S+)", out, re.MULTILINE):
        results[test] = "pass" if status == "PASSED" else "fail"
    return results, code, out


def _javascript_tests(paths):
    results, codes, outs = {}, [], []
    groups = {}
    for path in paths:
        groups.setdefault(_nearest(path, "package.json") or REPO, []).append(path)
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
    modules = {}
    for path in paths:
        module = _nearest(path, "go.mod") or REPO
        package = "./" + str((REPO / path).parent.relative_to(module))
        modules.setdefault(module, set()).add(package)
    results, codes, outs = {}, [], []
    for module, packages in sorted(modules.items()):
        code, out = run(["go", "test", "-json", "-count=1", *sorted(packages)], cwd=module)
        codes.append(code)
        outs.append(out)
        found = 0
        for line in out.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("Test") and event.get("Action") in ("pass", "fail", "skip"):
                results[f"{event['Package']}::{event['Test']}"] = event["Action"]
                found += 1
        if not found and code != 0:
            results.update({f"{module.name}/{p}::<package>": "fail" for p in packages})
    return results, codes, "\n".join(outs)


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


# Agent workspace, tools and grading.
GIT = ["git", "-c", "user.name=agent", "-c", "user.email=agent@localhost"]
IGNORED = (".venv/", "node_modules/", "__pycache__/", "*.pyc", ".pytest_cache/")
TOOL_SECONDS = 180
OUTPUT_CHARS = 4000
READ_LINES = 400


def workspace(task):
    _state(task, before=True)
    for path in task["test_paths"]:
        code, _ = run(["git", "checkout", "-q", task["parent"], "--", path])
        if code != 0:  # added by the commit
            (REPO / path).unlink(missing_ok=True)
    shutil.rmtree(REPO / ".git")
    shutil.rmtree("/work/mirror.git", ignore_errors=True)
    Path("/work/task.json").unlink(missing_ok=True)
    log = [run(["git", "init", "-q"])]
    (REPO / ".git/info/exclude").write_text("\n".join(IGNORED) + "\n")
    log.append(run(GIT + ["add", "-A"]))
    log.append(run(GIT + ["commit", "-q", "-m", "start"]))
    ok = all(code == 0 for code, _ in log)
    return {"stage": "ready" if ok else "workspace_error", "tail": log[-1][1][-1500:]}


def _clip(text, limit=OUTPUT_CHARS):
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - limit} characters omitted ...]\n{text[-half:]}"


def _path(arguments):
    path = (REPO / str(arguments.get("path", ""))).resolve()
    if path != REPO and REPO not in path.parents:
        raise ValueError("paths must stay inside the repository")
    return path


def tool(name, arguments):
    """One tool call's result text (errors are results the agent can read)."""
    try:
        if name == "bash":
            venv = REPO / ".venv"
            env = {"PATH": f"{venv}/bin:{os.environ['PATH']}"} if venv.exists() else {}
            code, out = run(str(arguments["command"]), seconds=TOOL_SECONDS, env=env)
            return f"exit code: {code}\n{_clip(out)}"
        path = _path(arguments)
        if name == "read_file":
            lines = path.read_text(errors="replace").splitlines()
            start = max(1, int(arguments.get("start_line") or 1))
            end = min(len(lines), int(arguments.get("end_line") or start + READ_LINES - 1))
            end = min(end, start + READ_LINES - 1)
            shown = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(start, end + 1))
            more = f"\n[{len(lines)} lines in total]" if end < len(lines) or start > 1 else ""
            return _clip(shown, 4 * OUTPUT_CHARS) + more
        if name == "write_file":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(arguments["content"]))
            return f"wrote {path.relative_to(REPO)}"
        if name == "edit_file":
            text = path.read_text()
            old, new = str(arguments["old"]), str(arguments["new"])
            count = text.count(old)
            if count != 1:
                return f"error: `old` occurs {count} times in {path.relative_to(REPO)}; it must occur once"
            path.write_text(text.replace(old, new))
            return f"edited {path.relative_to(REPO)}"
        return f"error: unknown tool {name}"
    except (KeyError, ValueError, OSError) as error:
        return f"error: {type(error).__name__}: {error}"


def patch():
    run(GIT + ["add", "-A"])
    code, out = run(["git", "diff", "--cached", "--binary", "HEAD"])
    return {"patch": out if code == 0 else "", "code": code}


def grade(task):
    _state(task, before=True)
    diff = Path("/work/patch.diff")
    applied = {"code": 0, "tail": ""}
    if diff.read_text().strip():
        excludes = [f"--exclude={p}" for p in task["test_paths"]]
        code, out = run(["git", "apply", "--whitespace=nowarn", *excludes, str(diff)])
        applied = {"code": code, "tail": out[-1500:]}
        if code != 0:
            return {"stage": "patch_failed", "apply": applied}
    results, code, log = TESTERS[task["language"]](task["test_paths"])
    return {
        "stage": "graded",
        "apply": applied,
        "results": results,
        "code": code,
        "tail": log[-TAIL:],
    }


if __name__ == "__main__":
    stage, argument = sys.argv[1], sys.argv[2]
    if stage == "tool":
        call = json.loads(sys.stdin.read())
        result = {"output": tool(call["name"], call["arguments"])}
    elif stage == "patch":
        result = patch()
    else:
        task = json.loads(Path(argument).read_text())
        result = {"setup": setup, "test": test, "workspace": workspace, "grade": grade}[stage](task)
    print("PR_TASK_RESULT " + json.dumps(result))
