"""PR -> task pilot stages on Modal (see `pr_tasks.py`).

Run from training/, in order:
    modal run -m model_hybrid.modal_pr_tasks::select [--count 1000] [--name pr-tasks-pilot]
    modal run --detach -m model_hybrid.modal_pr_tasks::verify [--limit N] [--workers 48]
    modal run --detach -m model_hybrid.modal_pr_tasks::issues

`select` reads the `changes` rows of the MIT collection at its current revision
(recorded), keeps single-language commits that change both source and test files
within `pr_tasks.LIMITS`, and writes `<name>/candidates.jsonl` (shapes only),
`<name>/selected.jsonl` (the seeded per-repository-capped sample with message,
patch and mirror path) and `<name>/selection.json` (counts per rejection reason and
language) to the `lm-exploration-hybrid-data` volume. It executes no repository
code.

`verify` gives each task two disposable sandboxes: the first checks out the commit
from the read-only mirror volume and installs dependencies (network on), its
filesystem snapshot then runs the commit's test files without network with source
files reverted to the parent and at the commit (`pr_task_runner.py`). A task is
verified when at least one test fails before and passes after and none regresses.
`issues` has Qwen3-Coder (vLLM, one H100) write the issue for every selected task
and flags answers that quote added source lines. Every call has `retries=0`, a
timeout of at most 6 hours and refuses work 24 hours after launch; sandboxes are
capped at one hour and at the launch deadline, and a supervisor cancels the spawned
call at that deadline.
"""

import base64
import json
import time
from pathlib import Path

import modal
from model_hybrid import modal_benchmarks

app = modal.App("lm-exploration-hybrid-pr-tasks")
app.include(modal_benchmarks.app)  # the deadline supervisor
volume = modal.Volume.from_name("lm-exploration-hybrid-data")
secret = modal.Secret.from_name("simple-llm-hf")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub==1.33.0")
    .env({"PYTHONPATH": "/project", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_dir("hybrid", "/project/model_hybrid", ignore=["__pycache__", "kernels"])
)
mirrors = modal.Volume.from_name("lm-exploration-pr-task-mirrors", create_if_missing=True)
# Repository code runs only in these sandboxes: no secrets, the mirror volume read-only,
# and the test runs without network.
sandbox_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ca-certificates", "build-essential", "xz-utils")
    .run_commands(
        "curl -fsSL https://nodejs.org/dist/v22.20.0/node-v22.20.0-linux-x64.tar.xz"
        " | tar -xJ -C /usr/local --strip-components=1",
        "corepack enable",
        "curl -fsSL https://go.dev/dl/go1.25.3.linux-amd64.tar.gz | tar -xz -C /usr/local",
        "pip install uv==0.9.5",
    )
    .env({"PATH": "/usr/local/go/bin:/root/go/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_file(Path(__file__).with_name("pr_task_runner.py"), "/runner/pr_task_runner.py")
)
teacher_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.30.0")
    .env({"PYTHONPATH": "/project", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_dir("hybrid", "/project/model_hybrid", ignore=["__pycache__", "kernels"])
)
TEACHER = ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "b2cff646eb4bb1d68355c01b18ae02e7cf42d120")
COLLECTION = "jfrick/mit-claude-codex-code-20261019"
SANDBOX_SECONDS = 60 * 60
LAUNCH_LIMIT_SECONDS = 24 * 60 * 60


def _shard_candidates(path):
    import gzip

    from model_hybrid import pr_tasks

    shard = path.split("/changes/", 1)[1]
    rejected, candidates = {}, []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            shape = pr_tasks.describe(row)
            if "reject" in shape:
                rejected[shape["reject"]] = rejected.get(shape["reject"], 0) + 1
                continue
            candidates.append(
                {
                    "repository": row["repository"],
                    "sha": row["sha"],
                    "parent": row["parent"],
                    "date": row["date"],
                    "shard": f"changes/{shard}",
                    **shape,
                }
            )
    return rejected, candidates


@app.function(
    image=image,
    cpu=32,
    memory=65536,
    timeout=2 * 60 * 60,
    retries=0,
    secrets=[secret],
    volumes={"/data": volume},
)
def select_candidates(name, count, per_repository, deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import gzip
    from concurrent.futures import ProcessPoolExecutor

    from huggingface_hub import HfApi, snapshot_download
    from model_hybrid import pr_tasks

    revision = HfApi().dataset_info(COLLECTION).sha
    local = snapshot_download(
        COLLECTION,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["changes/**", "repositories/**"],
        local_dir="/tmp/collection",
        max_workers=32,
    )
    shards = sorted(str(p) for p in Path(local, "changes").rglob("*.jsonl.gz"))
    rejected, candidates, seen = {}, [], set()
    with ProcessPoolExecutor(32) as pool:
        for shard_rejected, shard_candidates in pool.map(_shard_candidates, shards):
            for reason, n in shard_rejected.items():
                rejected[reason] = rejected.get(reason, 0) + n
            for candidate in shard_candidates:
                key = (candidate["repository"], candidate["sha"])
                if key in seen:
                    rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                    continue
                seen.add(key)
                candidates.append(candidate)
    mirrors = {}
    for path in sorted(Path(local, "repositories").rglob("*.git.tar.gz")):
        # Later collection runs sort last and win.
        mirrors[path.name.removesuffix(".git.tar.gz")] = str(path.relative_to(local))
    for candidate in candidates:
        candidate["mirror"] = mirrors.get(candidate["repository"].replace("/", "--"))
    chosen = pr_tasks.sample([c for c in candidates if c["mirror"]], count, per_repository)
    wanted = {(c["repository"], c["sha"]): c for c in chosen}
    for shard in sorted({c["shard"] for c in chosen}):
        with gzip.open(Path(local, shard), "rt") as handle:
            for line in handle:
                row = json.loads(line)
                if (item := wanted.get((row["repository"], row["sha"]))) is not None:
                    item["message"], item["patch"] = row["message"], row["patch"]
                    item["license"] = row["license_after"]["text"]
    out = Path("/data", name)
    out.mkdir(parents=True, exist_ok=True)
    for file, rows in (("candidates.jsonl", candidates), ("selected.jsonl", chosen)):
        with open(out / file, "w") as handle:
            handle.writelines(json.dumps(r) + "\n" for r in rows)

    def tally(rows, key):
        counts = {}
        for row in rows:
            counts[row[key]] = counts.get(row[key], 0) + 1
        return dict(sorted(counts.items()))

    summary = {
        "collection": COLLECTION,
        "revision": revision,
        "shards": len(shards),
        "rejected": dict(sorted(rejected.items())),
        "candidates": len(candidates),
        "candidates_by_language": tally(candidates, "language"),
        "candidate_repositories": len({c["repository"] for c in candidates}),
        "without_mirror": sum(1 for c in candidates if not c["mirror"]),
        "limits": pr_tasks.LIMITS,
        "per_repository": per_repository,
        "selected": len(chosen),
        "selected_by_language": tally(chosen, "language"),
        "selected_repositories": len({c["repository"] for c in chosen}),
    }
    (out / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary


def _sandbox_stage(sandbox, stage):
    process = sandbox.exec("python3", "/runner/pr_task_runner.py", stage, "/work/task.json")
    out = process.stdout.read()
    process.wait()
    for line in out.splitlines():
        if line.startswith("PR_TASK_RESULT "):
            return json.loads(line.split(" ", 1)[1])
    return {"stage": "runner_error", "tail": (out + process.stderr.read())[-3000:]}


def classify(before, after):
    """Fail-to-pass, pass-to-pass and regressed test names; absent before counts as failing."""
    names = sorted(after)
    return {
        "fail_to_pass": [t for t in names if after[t] == "pass" and before.get(t) != "pass"],
        "pass_to_pass": [t for t in names if after[t] == "pass" and before.get(t) == "pass"],
        "regressed": [t for t in sorted(before) if before[t] == "pass" and after.get(t) != "pass"],
    }


def _verify_one(task, deadline, record):
    seconds = int(min(SANDBOX_SECONDS, deadline - time.time()))
    if seconds < 600:
        return {"status": "deadline"}
    mirror = f"/mirrors/{Path(task['mirror']).name}"
    setup_box = modal.Sandbox.create(
        app=app,
        image=sandbox_image,
        timeout=seconds,
        cpu=2,
        memory=8192,
        volumes={"/mirrors": mirrors.read_only()},
        workdir="/work",
    )
    record(setup_box.object_id)
    try:
        setup_box.exec(
            "bash", "-c", f"mkdir -p /work/mirror.git && tar -xzf {mirror} -C /work/mirror.git"
        ).wait()
        fields = ("sha", "parent", "language", "source_paths", "test_paths")
        brief = base64.b64encode(json.dumps({k: task[k] for k in fields}).encode()).decode()
        setup_box.exec("bash", "-c", f"echo {brief} | base64 -d > /work/task.json").wait()
        setup = _sandbox_stage(setup_box, "setup")
        if setup.get("stage") != "installed":
            return {"status": setup.get("stage", "setup_error"), "setup": setup}
        snapshot = setup_box.snapshot_filesystem()
    finally:
        setup_box.terminate()
    seconds = int(min(SANDBOX_SECONDS, deadline - time.time()))
    if seconds < 300:
        return {"status": "deadline", "setup": setup}
    test_box = modal.Sandbox.create(
        app=app,
        image=snapshot,
        timeout=seconds,
        cpu=2,
        memory=8192,
        block_network=True,
        workdir="/work",
    )
    record(test_box.object_id)
    try:
        runs = _sandbox_stage(test_box, "test")
    finally:
        test_box.terminate()
    if "before" not in runs:
        return {"status": runs.get("stage", "test_error"), "setup": setup, "test": runs}
    split = classify(runs["before"]["results"], runs["after"]["results"])
    if not runs["after"]["results"]:
        status = "no_results"
    elif split["regressed"]:
        status = "regressed"
    elif not split["fail_to_pass"]:
        status = "no_fail_to_pass"
    else:
        status = "verified"
    return {"status": status, **split, "setup": setup, "test": runs}


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=6 * 60 * 60,
    retries=0,
    secrets=[secret],
    volumes={"/data": volume, "/mirrors": mirrors},
)
def verify_tasks(name, limit, workers, deadline):
    """Fetch the selected tasks' mirrors, then verify each task in its own sandboxes."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    folder = Path("/data", name)
    selection = json.loads((folder / "selection.json").read_text())
    lines = (folder / "selected.jsonl").read_text().splitlines()
    tasks = [json.loads(line) for line in lines]
    if limit:  # evenly spaced, so a small run covers every repository and language
        tasks = tasks[:: max(1, len(tasks) // limit)][:limit]
    for path in sorted({t["mirror"] for t in tasks}):
        target = Path("/mirrors", Path(path).name)
        if not target.exists():
            local = hf_hub_download(
                COLLECTION, path, repo_type="dataset", revision=selection["revision"]
            )
            target.write_bytes(Path(local).read_bytes())
    mirrors.commit()
    lock, sandboxes = threading.Lock(), []

    def record(sandbox_id):
        with lock:
            sandboxes.append(sandbox_id)
            (folder / "sandboxes.json").write_text(json.dumps(sandboxes) + "\n")

    def work(task):
        if time.time() >= deadline - 900:
            return {"status": "deadline"}
        try:
            return _verify_one(task, deadline, record)
        except Exception as error:  # noqa: BLE001 - one task's failure must not stop the rest
            return {"status": "error", "error": f"{type(error).__name__}: {error}"[:2000]}

    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        outcomes = list(pool.map(work, tasks))
    rows = [{**t, "verification": o} for t, o in zip(tasks, outcomes, strict=True)]
    suffix = f"-limit{limit}" if limit else ""
    with open(folder / f"verified{suffix}.jsonl", "w") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in rows)
    counts, by_language = {}, {}
    for row in rows:
        status = row["verification"]["status"]
        counts[status] = counts.get(status, 0) + 1
        language = by_language.setdefault(row["language"], {})
        language[status] = language.get(status, 0) + 1
    summary = {
        "tasks": len(rows),
        "status": dict(sorted(counts.items())),
        "by_language": by_language,
        "sandboxes": len(sandboxes),
        "seconds": round(time.time() - started),
    }
    (folder / f"verification{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary


@app.function(
    image=teacher_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=2 * 60 * 60,
    retries=0,
    secrets=[secret],
    volumes={"/data": volume},
)
def write_issues(name, deadline):
    """The teacher writes one issue per selected task; parse failures and leaks are recorded."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    from model_hybrid import pr_tasks
    from vllm import LLM, SamplingParams

    folder = Path("/data", name)
    tasks = [json.loads(line) for line in (folder / "selected.jsonl").read_text().splitlines()]
    model, revision = TEACHER
    llm = LLM(model=model, revision=revision, max_model_len=16384, gpu_memory_utilization=0.9)
    params = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=1024, seed=0)
    started = time.time()
    outputs = llm.chat([pr_tasks.issue_messages(t) for t in tasks], params)
    rows, counts = [], {}
    for task, output in zip(tasks, outputs, strict=True):
        text = output.outputs[0].text
        issue = pr_tasks.parse_issue(text)
        if issue is None:
            status = "unparsed"
        else:
            issue["leaked_lines"] = pr_tasks.leaks(issue, task["patch"], task["source_paths"])
            status = (
                "leaked"
                if issue["leaked_lines"]
                else ("ok" if issue["self_contained"] else "not_self_contained")
            )
        counts[status] = counts.get(status, 0) + 1
        rows.append(
            {
                "repository": task["repository"],
                "sha": task["sha"],
                "status": status,
                "issue": issue,
                "raw": None if issue else text[-2000:],
            }
        )
    with open(folder / "issues.jsonl", "w") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in rows)
    summary = {
        "teacher": {"model": model, "revision": revision},
        "tasks": len(rows),
        "status": dict(sorted(counts.items())),
        "seconds": round(time.time() - started),
    }
    (folder / "issues.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary


@app.local_entrypoint()
def issues(name: str = "pr-tasks-pilot"):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    call = write_issues.spawn(name, deadline)
    supervisor = modal_benchmarks.enforce_deadline.spawn([call.object_id], deadline)
    print(
        json.dumps(
            {"call": call.object_id, "supervisor": supervisor.object_id, "deadline_unix": deadline}
        )
    )


@app.local_entrypoint()
def verify(name: str = "pr-tasks-pilot", limit: int = 0, workers: int = 48):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    call = verify_tasks.spawn(name, limit, workers, deadline)
    supervisor = modal_benchmarks.enforce_deadline.spawn([call.object_id], deadline)
    print(
        json.dumps(
            {"call": call.object_id, "supervisor": supervisor.object_id, "deadline_unix": deadline}
        )
    )


@app.local_entrypoint()
def select(count: int = 1000, per_repository: int = 40, name: str = "pr-tasks-pilot"):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    print(json.dumps(select_candidates.remote(name, count, per_repository, deadline), indent=2))
