"""Grade Python function completions against hidden tests in an isolated sandbox.

Generations come from the sampling toolkit (`sampling.modal_sample`) with a recipe whose
`prompt_file` is `evaluation/python-tasks.json`. Each completion is cut at the end of the
prompted function and run by `execution_script` with CPU, memory, file and descriptor
limits, an empty environment and discarded output. It passes only if it prints a
per-case random marker after every hidden test. The reference solutions and a
network/environment isolation probe must pass the same harness before any model is scored.

Never run generated code outside a credential-free, network-blocked sandbox.
`modal_executor` provides one (`modal.Sandbox` with `block_network=True`, no secrets or
volumes and a bounded timeout); `subprocess_executor` is for reference solutions and tests.
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

LIMITS = {"cpu_seconds": 2, "address_space_bytes": 256 * 1024 * 1024, "file_bytes": 1024 * 1024}
CASE_TIMEOUT_SECONDS = 10
SANDBOX_TIMEOUT_SECONDS = 60 * 60
ISOLATION_PROBE = {
    "case_id": "isolation_probe",
    "code": (
        "import os, socket\nassert not os.environ\ntry:\n"
        "    s = socket.create_connection(('example.com', 443), timeout=1)\n"
        "except OSError:\n    blocked = True\nelse:\n    s.close()\n    blocked = False\n"
    ),
    "tests": ["assert blocked"],
    "test_imports": [],
}


def load_tasks(path):
    tasks = json.loads(Path(path).read_text())
    ids = [task["task_id"] for task in tasks["tasks"]]
    prompts = [task["prompt"] for task in tasks["tasks"]]
    if len(set(ids)) != len(ids) or len(set(prompts)) != len(prompts):
        raise ValueError("Task ids and prompts must be unique")
    return tasks


def function_completion(prompt, completion):
    """Prompt plus completion up to the first new top-level statement."""
    match = re.search(r"\n(?=\S)", completion)
    return prompt + (completion[: match.start()] if match else completion)


def reference_cases(tasks):
    """Reference solutions plus the isolation probe; all must pass before scoring models."""
    cases = [
        {
            "case_id": f"reference/{task['task_id']}",
            "task_id": task["task_id"],
            "code": task["reference"],
            "tests": task["tests"],
            "test_imports": task["test_imports"],
        }
        for task in tasks["tasks"]
    ]
    return [*cases, dict(ISOLATION_PROBE)]


def sample_cases(tasks, result):
    """One case per sampling-toolkit sample whose prompt is a task prompt."""
    by_prompt = {task["prompt"]: task for task in tasks["tasks"]}
    cases = []
    for model in result["models"]:
        for sample in model["samples"]:
            task = by_prompt.get(sample["prompt"])
            if task is None:
                continue
            code = function_completion(sample["prompt"], sample["completion"])
            cases.append(
                {
                    "case_id": f"{model['name']}/{sample['mode']}/{sample['seed']}/{task['task_id']}",
                    "model": model["name"],
                    "mode": sample["mode"],
                    "seed": sample["seed"],
                    "task_id": task["task_id"],
                    "code": code,
                    "tests": task["tests"],
                    "test_imports": task["test_imports"],
                }
            )
    return cases


def syntax_valid(code):
    try:
        ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return True


def execution_script(case, marker):
    """Program that prints only `marker` when the candidate passes every hidden test."""
    payload = {"code": case["code"], "tests": case["tests"], "imports": case["test_imports"]}
    return f"""
import os, resource
resource.setrlimit(resource.RLIMIT_CPU, ({LIMITS["cpu_seconds"]}, {LIMITS["cpu_seconds"]}))
resource.setrlimit(resource.RLIMIT_AS, ({LIMITS["address_space_bytes"]}, {LIMITS["address_space_bytes"]}))
resource.setrlimit(resource.RLIMIT_FSIZE, ({LIMITS["file_bytes"]}, {LIMITS["file_bytes"]}))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
os.environ.clear()
result_fd = os.dup(1)
null = os.open('/dev/null', os.O_WRONLY)
os.dup2(null, 1)
os.dup2(null, 2)
payload = {payload!r}
namespace = {{'__name__': '__main__'}}
try:
    for statement in payload['imports']:
        exec(compile(statement, '<setup>', 'exec'), namespace)
    exec(compile(payload['code'], '<candidate>', 'exec'), namespace)
    for statement in payload['tests']:
        exec(compile(statement, '<hidden-test>', 'exec'), namespace)
except BaseException:
    os.write(result_fd, b'FAILED')
else:
    os.write(result_fd, {marker.encode()!r})
"""


def grade(cases, execute):
    """Run every case; `execute(script)` returns (returncode, stdout) from a sandbox."""
    results = []
    for case in cases:
        marker = uuid.uuid4().hex
        code, stdout = execute(execution_script(case, marker))
        results.append(
            {
                **{k: v for k, v in case.items() if k not in ("tests", "test_imports")},
                "syntax_valid": syntax_valid(case["code"]),
                "passed": code == 0 and stdout == marker,
            }
        )
    return results


def summarize(results):
    """Pass and syntax-valid fractions per model and decoding mode."""
    groups = {}
    for row in results:
        groups.setdefault((row["model"], row["mode"]), []).append(row)
    return [
        {
            "model": model,
            "mode": mode,
            "cases": len(rows),
            "passed": sum(r["passed"] for r in rows),
            "pass_fraction": sum(r["passed"] for r in rows) / len(rows),
            "syntax_valid_fraction": sum(r["syntax_valid"] for r in rows) / len(rows),
        }
        for (model, mode), rows in sorted(groups.items())
    ]


def subprocess_executor(script):
    """Local execution: only for reference solutions and tests, never model output."""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        capture_output=True,
        text=True,
        timeout=CASE_TIMEOUT_SECONDS,
        check=False,
    )
    return result.returncode, result.stdout


@contextmanager
def modal_executor():
    """One network-blocked Modal sandbox without secrets or volumes, one process per case."""
    import modal

    app = modal.App.lookup("python-evaluation-sandbox", create_if_missing=True)
    sandbox = modal.Sandbox.create(
        "sleep",
        str(SANDBOX_TIMEOUT_SECONDS),
        app=app,
        image=modal.Image.debian_slim(python_version="3.12"),
        block_network=True,
        timeout=SANDBOX_TIMEOUT_SECONDS,
        cpu=1,
        memory=1024,
    )

    def execute(script):
        process = sandbox.exec("python", "-I", "-S", "-c", script, timeout=CASE_TIMEOUT_SECONDS)
        process.wait()
        return process.returncode, process.stdout.read()

    try:
        yield execute
    finally:
        sandbox.terminate()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Grade sampled Python task completions.")
    parser.add_argument("--tasks", default="evaluation/python-tasks.json")
    parser.add_argument("--samples", required=True, help="sampling.modal_sample result JSON")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    tasks = load_tasks(args.tasks)
    cases = sample_cases(tasks, json.loads(Path(args.samples).read_text()))
    if not cases:
        raise ValueError("No samples use the task prompts")
    with modal_executor() as execute:
        references = grade(reference_cases(tasks), execute)
        failed = [r["case_id"] for r in references if not r["passed"]]
        if failed:
            raise RuntimeError(f"Harness check failed for {failed}")
        results = grade(cases, execute)
    report = {
        "tasks": {k: v for k, v in tasks.items() if k not in ("tasks", "reference_hashes")},
        "limits": LIMITS,
        "references_passed": len(references),
        "summary": summarize(results),
        "cases": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
