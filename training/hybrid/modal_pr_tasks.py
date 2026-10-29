"""PR -> task pilot stages on Modal (see `pr_tasks.py`).

Run from training/, in order:
    modal run -m model_hybrid.modal_pr_tasks::select [--count 1000] [--name pr-tasks-pilot]
    modal run --detach -m model_hybrid.modal_pr_tasks::verify [--limit N] [--workers 48]
    modal run --detach -m model_hybrid.modal_pr_tasks::issues [--limit N]

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
`issues` has GLM-5.3 (through OpenRouter, with the launching shell's
`OPENROUTER_API_KEY`) draft the issue of every selected task from the commit message
and the test changes only (never the source changes), up to four drafts until one
parses, quotes no added source line, names no new internal code the tests never use
and does not mention tests or the commit. Every call has `retries=0`, a
timeout of at most 6 hours and refuses work 24 hours after launch; sandboxes are
capped at one hour and at the launch deadline, and a supervisor cancels the spawned
call at that deadline.
"""

import base64
import json
import os
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
# The key comes from the launching shell's OPENROUTER_API_KEY and lives only for the run.
openrouter = modal.Secret.from_dict(
    {"OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", "")}
)
DRAFTS = 4  # teacher drafts per task, requested until one is clean
# GLM-5.3 (zai-org/GLM-5.3, MIT-style license) through OpenRouter; outputs may train models.
TEACHER = "z-ai/glm-5.3"
TEACHER_WORKERS = 32
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


def sandbox_stage(sandbox, stage, argument="/work/task.json", stdin=None):
    """The runner's PR_TASK_RESULT for one stage (`stdin` text is sent to it)."""
    process = sandbox.exec("python3", "/runner/pr_task_runner.py", stage, argument)
    if stdin is not None:
        process.stdin.write(stdin.encode())
        process.stdin.write_eof()
        process.stdin.drain()
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


def installed_snapshot(sandbox_app, task, deadline, record):
    """(filesystem snapshot, setup result) of the task's repository checked out at the
    commit with dependencies installed; the snapshot is None when setup failed."""
    seconds = int(min(SANDBOX_SECONDS, deadline - time.time()))
    if seconds < 600:
        return None, {"stage": "deadline"}
    mirror = f"/mirrors/{Path(task['mirror']).name}"
    setup_box = modal.Sandbox.create(
        app=sandbox_app,
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
        setup = sandbox_stage(setup_box, "setup")
        if setup.get("stage") != "installed":
            return None, setup
        return setup_box.snapshot_filesystem(), setup
    finally:
        setup_box.terminate()


def _verify_one(task, deadline, record):
    snapshot, setup = installed_snapshot(app, task, deadline, record)
    if snapshot is None:
        return {"status": setup.get("stage", "setup_error"), "setup": setup}
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
        runs = sandbox_stage(test_box, "test")
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


def _teacher_draft(messages):
    """One teacher answer and its token usage, through OpenRouter's chat API."""
    import urllib.request

    body = {
        "model": TEACHER,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096,
        "reasoning": {"effort": "low", "exclude": True},
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read())
    text = (data["choices"][0]["message"].get("content") or "") if data.get("choices") else ""
    return text, data.get("usage", {}), data.get("provider")


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=2 * 60 * 60,
    retries=0,
    secrets=[openrouter],
    volumes={"/data": volume},
)
def write_issues(name, deadline, limit=0):
    """The teacher drafts each task's issue until one is clean (at most DRAFTS drafts)."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Launch with OPENROUTER_API_KEY set")
    import urllib.error
    from concurrent.futures import ThreadPoolExecutor

    from model_hybrid import pr_tasks

    folder = Path("/data", name)
    tasks = [json.loads(line) for line in (folder / "selected.jsonl").read_text().splitlines()]
    if limit:
        tasks = tasks[:: max(1, len(tasks) // limit)][:limit]

    def write(task):
        messages = pr_tasks.issue_messages(task)
        drafts, usage, providers, best = [], {"prompt_tokens": 0, "completion_tokens": 0}, [], None
        for _ in range(DRAFTS):
            if time.time() >= deadline - 300:
                break
            try:
                text, used, provider = _teacher_draft(messages)
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as error:
                drafts.append(f"<request failed: {type(error).__name__}: {error}>"[:500])
                continue
            for key in usage:
                usage[key] += int(used.get(key) or 0)
            providers.append(provider)
            drafts.append(text)
            best = pr_tasks.best_issue(drafts, task)
            if best[0] == "ok":
                break
        status, issue, text = best or ("unparsed", None, drafts[-1] if drafts else "")
        return {
            "repository": task["repository"],
            "sha": task["sha"],
            "status": status,
            "issue": issue,
            "drafts": len(drafts),
            "usage": usage,
            "providers": sorted({p for p in providers if p}),
            "raw": None if issue else text[-2000:],
        }

    started = time.time()
    with ThreadPoolExecutor(TEACHER_WORKERS) as pool:
        rows = list(pool.map(write, tasks))
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    suffix = f"-limit{limit}" if limit else ""
    with open(folder / f"issues{suffix}.jsonl", "w") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in rows)
    summary = {
        "teacher": {"model": TEACHER, "via": "openrouter"},
        "tasks": len(rows),
        "status": dict(sorted(counts.items())),
        "drafts": sum(r["drafts"] for r in rows),
        "tokens": {
            k: sum(r["usage"][k] for r in rows) for k in ("prompt_tokens", "completion_tokens")
        },
        "seconds": round(time.time() - started),
    }
    (folder / f"issues{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary


@app.local_entrypoint()
def issues(name: str = "pr-tasks-pilot", limit: int = 0):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    call = write_issues.spawn(name, deadline, limit)
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
