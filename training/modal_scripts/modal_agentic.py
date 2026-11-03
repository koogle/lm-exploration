"""Agent episodes on verified PR tasks (see `data/pr_task_agent.py`): teacher trajectories and evaluation.

Run from training/:
    modal run --detach -m modal_scripts.modal_agentic::solve [--limit N] [--workers 32]
    modal run --detach -m modal_scripts.modal_agentic::evaluate --export <run>/<export> [--limit N]
        [--agent-format tools|open-swe] [--temperature T] [--max-turns N]
    modal run -m modal_scripts.modal_agentic::prepare [--recipe data/recipes/agentic-sft-v1.json]
    modal run --detach -m modal_scripts.modal_agentic::reference_train --config <run config>
        --run <name> [--hours 20]
    modal run --detach -m modal_scripts.modal_agentic::reference_evaluate --models M1,sft:<run>
        [--agent-format qwen-tools|mini-swe|open-swe] [--temperature T] [--max-turns N]
    modal run --detach -m modal_scripts.modal_agentic::reference_bench --model sft:<run>

Tasks are the pilot's verified tasks with a clean issue (`pr-tasks-pilot/verified.jsonl`
and `issues.jsonl` on the `lm-exploration-hybrid-data` volume). Each episode installs
the repository in a sandbox with network (`modal_pr_tasks.installed_snapshot`), then
works in a sandbox from that snapshot without network, and is graded in a third one.

`solve` has GLM-5.3 (through OpenRouter, with the launching shell's
`OPENROUTER_API_KEY`) work every task with the tools (training repositories'
tasks `attempts` times, held-out ones once) and writes
`<name>/agentic/<run>/trajectories.jsonl` (messages, patch, grade, resolved, token
usage, held-out flag) and `summary.json`. `evaluate` runs a hybrid export (GLM chat format, the
harness's tools or Open-SWE's bash tool, greedy or sampled) on the held-out
repositories' tasks in one H100 call and writes
`<name>/agentic/<run>/` the same way. `prepare` turns the recipe's `trajectories` run
into the chat SFT subset `agentic` (resolved episodes from training repositories
only, rendered with `chat_sft.prepare_subset`).

The reference stages run Qwen3 models through the same tasks and data, to separate
the effect of broad pretraining from the agent data. `reference_train` fully
fine-tunes a pinned Qwen3 base with `trainer.reference` (the shared training loop, on
eight GPUs) on a chat SFT recipe prepared by `modal_agent_mix.py`, into
`lm-exploration-hybrid-runs/agentic-reference/<run>/`: the loop's checkpoints and
metrics, and bf16 exports under `export/` (`final` at the end). Launching it again for
the same run continues from the last checkpoint. `reference_evaluate` runs pinned
models (Qwen3 chat models with thinking off, mini-coder) or those exports
(`sft:<run>[@<export>]`) through vLLM on the held-out tasks, in one agent format,
greedy or sampled, and writes the same files as `evaluate`. `reference_bench` runs
Terminal-Bench 2.1 (Harbor, Terminus 2, Modal sandboxes) against an export served by
vLLM in the same container. Every call has `retries=0` and a timeout under 24 hours,
refuses work after the launch deadline, and a supervisor cancels the spawned calls
at the deadline; sandboxes are capped at the deadline.
"""

import json
import os
import threading
import time
from pathlib import Path

import modal
from model_hybrid import modal_benchmarks
from model_hybrid.image import build

from modal_scripts import modal_chat_sft_data, modal_pr_tasks
from workflows.images import add_project

app = modal.App("lm-exploration-hybrid-agentic")
app.include(modal_benchmarks.app)  # the deadline supervisor
volume = modal_pr_tasks.volume
runs_volume = modal.Volume.from_name("lm-exploration-hybrid-runs")
gpu_image = build(("jinja2==3.1.6",))
LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
TEACHER_MAX_TOKENS = 8192
EVAL_SANDBOX_SECONDS = 7 * 60 * 60  # long-turn agent evaluations run for hours

# Reference models: Qwen3 with its broad pretraining, through the same tasks and data.
REFERENCE_MODELS = {
    "Qwen/Qwen3-0.6B-Base": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    "Qwen/Qwen3-1.7B-Base": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    # Qwen3-1.7B trained on the 400k mini-coder trajectories (mini-swe-agent format).
    "ricdomolm/mini-coder-1.7b": "5b7fbc90b6c4684c546de7f811865cff4414a96c",
}
AGENT_FORMATS = ("qwen-tools", "mini-swe", "open-swe")
# Qwen3's recommended non-thinking sampling, for runs that sample instead of decoding greedily.
SAMPLING = {"top_p": 0.8, "top_k": 20}
REFERENCE_FOLDER = "agentic-reference"  # on the runs volume
REFERENCE_CONTEXT = 32768  # the pinned Qwen3 models' positions
# One launch of `trainer.reference`; a longer run continues from its checkpoint.
REFERENCE_TRAIN_SECONDS = 20 * 60 * 60
# As the hybrid evaluation (agent_policy.LocalPolicy): greedy, 1,536 new tokens per turn.
REFERENCE_MAX_NEW_TOKENS = 1536
# vLLM's FlashInfer kernels compile at load time, so the image carries nvcc (as in
# modal_teacher.py); training uses the torch and transformers that vLLM installs.
reference_base = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.31.0", "huggingface-hub[hf_transfer]", "pyarrow==26.0.0")
    .env(
        {
            "HF_HOME": "/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/project",
            "CUDA_HOME": "/usr/local/cuda",
        }
    )
)
reference_image = add_project(reference_base)
# Training adds Liger's Triton kernels (`trainer.reference`).
train_image = add_project(reference_base.pip_install("liger-kernel==0.8.4"))
# Harbor (the Terminal-Bench runner) in its own tool environment, next to vLLM.
terminal_bench_image = add_project(
    reference_base.run_commands(
        "pip install uv==0.9.5",
        "uv tool install 'harbor[modal]==0.25.0' --with 'modal[api-proxy-support]'",
    ).env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin"})
)
hf_cache = modal.Volume.from_name("lm-exploration-hf-cache", create_if_missing=True)


def load_tasks(folder):
    """Verified tasks with a clean issue, each with its issue and held-out flag."""
    from training_data import pr_task_agent as agent

    issues = {}
    for line in (folder / "issues.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["status"] == "ok":
            issues[(row["repository"], row["sha"])] = row["issue"]
    tasks = []
    for line in (folder / "verified.jsonl").read_text().splitlines():
        task = json.loads(line)
        issue = issues.get((task["repository"], task["sha"]))
        if task["verification"]["status"] == "verified" and issue:
            tasks.append({**task, "issue": issue, "heldout": agent.heldout(task["repository"])})
    return tasks


class Workspace:
    """The agent's sandbox for one task, from the installed snapshot, without network."""

    def __init__(self, task, snapshot, deadline, record, lifetime=modal_pr_tasks.SANDBOX_SECONDS):
        self.task, self.snapshot, self.deadline, self.record = task, snapshot, deadline, record
        self.lifetime = lifetime
        self.box = self._sandbox()
        started = modal_pr_tasks.sandbox_stage(self.box, "workspace")
        if started.get("stage") != "ready":
            self.close()
            raise RuntimeError(f"workspace: {json.dumps(started)[:1000]}")

    def _sandbox(self):
        seconds = int(min(self.lifetime, self.deadline - time.time()))
        if seconds < 300:
            raise TimeoutError("launch deadline")
        box = modal.Sandbox.create(
            app=app,
            image=self.snapshot,
            timeout=seconds,
            cpu=2,
            memory=8192,
            block_network=True,
            workdir="/work/repo",
        )
        self.record(box.object_id)
        return box

    def call(self, name, arguments):
        call = json.dumps({"name": name, "arguments": arguments})
        result = modal_pr_tasks.sandbox_stage(self.box, "tool", "-", stdin=call)
        return result.get("output", f"error: {json.dumps(result)[:2000]}")

    def finish(self):
        """The agent's patch and its grade against the hidden tests."""
        patch = modal_pr_tasks.sandbox_stage(self.box, "patch", "-").get("patch", "")
        self.close()
        grader = self._sandbox()
        try:
            writer = grader.exec("bash", "-c", "cat > /work/patch.diff")
            writer.stdin.write(patch.encode())
            writer.stdin.write_eof()
            writer.stdin.drain()
            writer.wait()
            grade = modal_pr_tasks.sandbox_stage(grader, "grade")
        finally:
            grader.terminate()
        return patch, grade

    def close(self):
        self.box.terminate()


def _teacher_turn(messages, usage):
    """GLM-5.3's next assistant message through OpenRouter (three attempts)."""
    import urllib.error
    import urllib.request

    from training_data import pr_task_agent as agent

    body = {
        "model": modal_pr_tasks.TEACHER,
        "messages": messages,
        "tools": agent.TOOLS,
        "temperature": 0.6,
        "max_tokens": TEACHER_MAX_TOKENS,
        "reasoning": {"effort": "medium", "exclude": True},
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.loads(response.read())
            message = data["choices"][0]["message"]
            break
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError):
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))
    for key in ("prompt_tokens", "completion_tokens"):
        usage[key] = usage.get(key, 0) + int(data.get("usage", {}).get(key) or 0)
    usage["cost"] = usage.get("cost", 0) + float(data.get("usage", {}).get("cost") or 0)
    reply = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        reply["tool_calls"] = [
            {"id": c.get("id"), "type": "function", "function": c["function"]}
            for c in message["tool_calls"]
        ]
    return reply


def _summary(rows, started, max_turns=None):
    from training_data import pr_task_agent as agent

    def rate(selected):
        tasks = {(r["repository"], r["sha"]) for r in selected}
        solved = {(r["repository"], r["sha"]) for r in selected if r["resolved"]}
        return {
            "episodes": len(selected),
            "resolved": sum(r["resolved"] for r in selected),
            "rate": round(sum(r["resolved"] for r in selected) / max(1, len(selected)), 4),
            "tasks": len(tasks),
            "tasks_resolved": len(solved),
        }

    ends, by_language = {}, {}
    for row in rows:
        ends[row["end"]] = ends.get(row["end"], 0) + 1
        by_language.setdefault(row["language"], []).append(row)
    usage = {
        k: sum(r["usage"].get(k, 0) for r in rows)
        for k in ("prompt_tokens", "completion_tokens", "cost")
    }
    return {
        "all": rate(rows),
        "heldout": rate([r for r in rows if r["heldout"]]),
        "train": rate([r for r in rows if not r["heldout"]]),
        "by_language": {k: rate(v) for k, v in sorted(by_language.items())},
        "ends": dict(sorted(ends.items())),
        "turns_mean": round(sum(r["turns"] for r in rows) / max(1, len(rows)), 2),
        "usage": usage,
        "max_turns": max_turns or agent.MAX_TURNS,
        "seconds": round(time.time() - started),
    }


def _write(folder, rows, summary):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "trajectories.jsonl", "w") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in rows)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()


def _recorder(folder):
    lock, sandboxes = threading.Lock(), []

    def record(sandbox_id):
        with lock:
            sandboxes.append(sandbox_id)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "sandboxes.json").write_text(json.dumps(sandboxes) + "\n")

    return record


def _row(task, episode, patch, grade, usage):
    from training_data import pr_task_agent as agent

    return {
        "repository": task["repository"],
        "sha": task["sha"],
        "attempt": task.get("attempt", 0),
        "language": task["language"],
        "heldout": task["heldout"],
        "end": episode.get("end", "error"),
        "turns": episode.get("turns", 0),
        "resolved": agent.resolved(grade, task["verification"]) if grade else False,
        "grade": {k: v for k, v in (grade or {}).items() if k != "tail"},
        "patch": patch,
        "messages": episode.get("messages", []),
        "usage": usage,
        "error": episode.get("error"),
    }


def _select(tasks, limit, heldout_only=False):
    if heldout_only:
        tasks = [t for t in tasks if t["heldout"]]
    if limit:  # evenly spaced, so a small run covers every repository and language
        tasks = tasks[:: max(1, len(tasks) // limit)][:limit]
    return tasks


@app.function(
    image=modal_pr_tasks.image,
    cpu=4,
    memory=16384,
    timeout=6 * 60 * 60,
    retries=0,
    secrets=[modal_pr_tasks.openrouter],
    volumes={"/data": volume},
)
def solve_tasks(name, run, limit, workers, attempts, deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Launch with OPENROUTER_API_KEY set")
    from concurrent.futures import ThreadPoolExecutor

    from training_data import pr_task_agent as agent

    folder = Path("/data", name)
    out = folder / "agentic" / run
    tasks = [
        {**task, "attempt": attempt}
        for task in _select(load_tasks(folder), limit)
        for attempt in range(1 if task["heldout"] else attempts)
    ]
    record = _recorder(out)

    def work(task):
        episode, usage, patch, grade = (
            {"messages": agent.first_messages(task, task["issue"])},
            {},
            "",
            None,
        )
        try:
            if time.time() >= deadline - 1800:
                raise TimeoutError("launch deadline")
            snapshot, setup = modal_pr_tasks.installed_snapshot(app, task, deadline, record)
            if snapshot is None:
                raise RuntimeError(f"setup: {setup.get('stage')}")
            workspace = Workspace(task, snapshot, deadline, record)
            try:
                agent.run_episodes(
                    [episode],
                    lambda conversations: [_teacher_turn(m, usage) for m in conversations],
                    lambda _, name, arguments: workspace.call(name, arguments),
                    workers=4,
                )
            except BaseException:
                workspace.close()
                raise
            patch, grade = workspace.finish()
        except Exception as error:  # noqa: BLE001 - one task's failure must not stop the rest
            episode["error"] = f"{type(error).__name__}: {error}"[:2000]
        return _row(task, episode, patch, grade, usage)

    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(work, tasks))
    summary = {"teacher": modal_pr_tasks.TEACHER, "run": run, **_summary(rows, started)}
    _write(out, rows, summary)
    return summary


@app.function(
    image=gpu_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=EVAL_SANDBOX_SECONDS + 60 * 60,
    retries=0,
    volumes={"/data": volume, "/runs": runs_volume},
)
def evaluate_export(
    export, agent_format, temperature, name, run, limit, workers, max_turns, deadline
):
    """A hybrid export as the agent on the held-out tasks, in `agent_format` ("tools" or
    "open-swe", `model_hybrid.agent_policy`), greedy at temperature 0, otherwise sampled
    with the reference evaluation's top-p and top-k."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import torch
    from model_hybrid import agent_policy, generate
    from training_data import open_swe_format
    from training_data import pr_task_agent as agent

    from tokenizer import glm

    run_name, export_name = export.split("/")
    lm = generate.load_export(
        Path("/runs", run_name, "export", export_name), dtype=torch.bfloat16, device="cuda"
    )
    sampling = SAMPLING if temperature > 0 else {}
    local = agent_policy.LocalPolicy(
        lm, glm.load(), rows=96, agent_format=agent_format, temperature=temperature, **sampling
    )
    header = {
        "export": export,
        "agent_format": agent_format,
        "context": lm.config.block_size,
        "sampling": {"temperature": temperature, **sampling},
        "max_turns": max_turns or agent.MAX_TURNS,
    }
    # As in `evaluate_reference`: Open-SWE's instructions work in /testbed.
    setup_command = (
        f"cd / && mv /work/repo {open_swe_format.WORKDIR} && ln -s {open_swe_format.WORKDIR} /work/repo"
        if agent_format == "open-swe"
        else None
    )
    return _evaluate_heldout(
        local, header, name, run, limit, workers, deadline, setup_command, max_turns
    )


def _call(workspace, name, arguments):
    """A tool call whose sandbox failure becomes the tool's result, so one lost sandbox
    ends its own episode's progress instead of the whole evaluation."""
    try:
        return workspace.call(name, arguments)
    except Exception as error:  # noqa: BLE001
        return f"error: sandbox failure: {type(error).__name__}: {error}"[:2000]


def _evaluate_heldout(
    local, header, name, run, limit, workers, deadline, setup_command=None, max_turns=None
):
    """Run `local` (a batch policy) as the agent on the held-out repositories' tasks;
    `setup_command` is a bash command run in each workspace before the episode, and
    `max_turns` overrides `pr_task_agent.MAX_TURNS`."""
    from concurrent.futures import ThreadPoolExecutor

    from training_data import pr_task_agent as agent

    folder = Path("/data", name)
    out = folder / "agentic" / run
    tasks = _select(load_tasks(folder), limit, heldout_only=True)
    record = _recorder(out)
    turn = [0]

    def policy(conversations):
        replies = local(conversations)
        turn[0] += 1
        progress = {
            "turn": turn[0],
            "active": len(conversations),
            "seconds": round(time.time() - started),
        }
        print(json.dumps(progress), flush=True)
        return replies

    def start(task):
        try:
            snapshot, setup = modal_pr_tasks.installed_snapshot(app, task, deadline, record)
            if snapshot is None:
                return None, f"setup: {setup.get('stage')}"
            # Every episode advances in lockstep with the batch, so sandboxes live longer.
            workspace = Workspace(task, snapshot, deadline, record, EVAL_SANDBOX_SECONDS)
            if setup_command:
                result = workspace.call("bash", {"command": setup_command})
                if not result.startswith("exit code: 0"):
                    workspace.close()
                    return None, f"setup command: {result[:500]}"
            return workspace, None
        except Exception as error:  # noqa: BLE001
            return None, f"{type(error).__name__}: {error}"[:2000]

    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        opened = list(pool.map(start, tasks))
    episodes = [
        {"messages": agent.first_messages(t, t["issue"]), "workspace": w, "error": e}
        for t, (w, e) in zip(tasks, opened, strict=True)
    ]
    live = [e for e in episodes if e["workspace"]]
    agent.run_episodes(
        live,
        policy,
        lambda episode, name, arguments: _call(episode["workspace"], name, arguments),
        max_turns=max_turns or agent.MAX_TURNS,
        workers=workers,
    )

    def finish(episode):
        if not episode["workspace"]:
            return "", None
        try:
            return episode["workspace"].finish()
        except Exception as error:  # noqa: BLE001
            episode["error"] = f"{type(error).__name__}: {error}"[:2000]
            return "", None

    with ThreadPoolExecutor(workers) as pool:
        finished = list(pool.map(finish, episodes))
    rows = [
        _row(t, e, patch, grade, {})
        for t, e, (patch, grade) in zip(tasks, episodes, finished, strict=True)
    ]
    summary = {**header, "run": run, **_summary(rows, started, max_turns)}
    _write(out, rows, summary)
    return summary


@app.function(
    image=modal_chat_sft_data.image,
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    retries=0,
    secrets=[modal_chat_sft_data.secret],
    volumes={"/data": volume},
)
def prepare_trajectories(recipe, deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import numpy as np
    from training_data import chat_sft
    from training_data import pr_task_agent as agent

    from tokenizer import glm as tokenizer

    lines = Path("/data", recipe["trajectories"]).read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    kept = [r for r in rows if r["resolved"] and not r["heldout"]]
    screen = np.load(Path("/data", recipe["screen"]))
    glm = tokenizer.load(tokenizer.tokenizer_file("/tmp/tokenizer"))
    meta = chat_sft.prepare_subset(
        recipe,
        "agentic",
        kept,
        Path("/data", recipe["name"]),
        glm,
        screen,
        converter=lambda row: agent.training_row(row["messages"]),
    )
    meta["episodes"] = {"all": len(rows), "resolved_training_repositories": len(kept)}
    volume.commit()
    return meta


@app.function(
    image=train_image,
    gpu="H100:8",
    cpu=32,
    memory=262144,
    timeout=REFERENCE_TRAIN_SECONDS,
    retries=0,
    volumes={"/data": volume, "/runs": runs_volume, "/hf": hf_cache},
)
def train_reference(config, run, seconds, deadline):
    """`trainer.reference` with run config `config` (a path under training/) over eight
    GPUs into `agentic-reference/<run>/`, for at most `seconds`; a run folder with a
    checkpoint continues from it (the loop's `latest.json`)."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import subprocess

    volume.reload()
    runs_volume.reload()
    stop = min(deadline - 1800, time.time() + min(seconds, REFERENCE_TRAIN_SECONDS) - 1800)
    out = Path("/runs", REFERENCE_FOLDER, run)
    command = [
        "python", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
        "-m", "trainer.reference", "--config", str(Path("/project", config)),
        "--run-dir", str(out), "--data-root", "/data", "--stop-unix", str(stop),
    ]  # fmt: skip
    committing, done = threading.Event(), threading.Event()

    def commit():  # checkpoints and exports survive a lost container
        while not done.wait(600):
            runs_volume.commit()
        committing.set()

    threading.Thread(target=commit, daemon=True).start()
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    try:
        subprocess.run(command, cwd="/project", env=env, check=True)
    finally:
        done.set()
        committing.wait(30)
        runs_volume.commit()
    return json.loads((out / "report.json").read_text())


def _reference_export(model):
    """(folder, run.json) of `sft:<run>` (its final export) or `sft:<run>@<export>`."""
    runs_volume.reload()
    run, _, export = model.removeprefix("sft:").partition("@")
    folder = Path("/runs", REFERENCE_FOLDER, run, "export", export or "final")
    return folder, json.loads((folder / "run.json").read_text())


@app.function(
    image=reference_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=8 * 60 * 60,
    retries=0,
    volumes={"/data": volume, "/runs": runs_volume, "/hf": hf_cache},
)
def evaluate_reference(
    model, agent_format, temperature, name, run, limit, workers, max_turns, deadline
):
    """A reference model as the agent on the held-out tasks, through vLLM.

    `model` is a pinned model (Qwen3 chat models zero-shot with thinking off, or
    mini-coder) or `sft:<run>[@<export>]`, a `train_reference` export. `agent_format` is
    "qwen-tools" (Qwen3's tool format; chat models get the empty thinking block) or
    "mini-swe" (bash-only, `training_data.mini_swe_format`) or "open-swe" (Open-SWE-Traces'
    bash tool, `training_data.open_swe_format`). Decoding is greedy at
    temperature 0, otherwise sampled with Qwen3's recommended top-p and top-k.
    """
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    from training_data import mini_swe_format, open_swe_format, qwen_agent_chat
    from training_data import pr_task_agent as agent
    from vllm import LLM, SamplingParams

    if agent_format not in AGENT_FORMATS:
        raise ValueError(f"Unknown agent format {agent_format}")
    context = REFERENCE_CONTEXT
    if model.startswith("sft:"):
        folder, trained = _reference_export(model)
        source, revision, chat = str(folder), None, False
        header = {"model": model, "base": trained["base"], "base_revision": trained["revision"]}
        # The export's config covers the context the run was trained at.
        context = trained["context"]
    else:
        source, revision = model, REFERENCE_MODELS[model]
        chat = agent_format == "qwen-tools"
        header = {"model": model, "revision": revision}
    llm = LLM(
        model=source,
        revision=revision,
        max_model_len=context,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        temperature=temperature,
        **(SAMPLING if temperature > 0 else {}),
        max_tokens=REFERENCE_MAX_NEW_TOKENS,
        stop_token_ids=[tokenizer.convert_tokens_to_ids(qwen_agent_chat.TURN_END)],
        skip_special_tokens=False,
    )

    def prompt_text(messages):
        if agent_format == "open-swe":
            return qwen_agent_chat.render(
                open_swe_format.conversation(messages),
                open_swe_format.TOOLS,
                add_generation_prompt=True,
                chat=chat,
            )
        if agent_format == "mini-swe":
            return mini_swe_format.render(
                mini_swe_format.conversation(messages), add_generation_prompt=True
            )
        return qwen_agent_chat.render(
            agent.training_row(messages)["messages"],
            agent.TOOLS,
            add_generation_prompt=True,
            chat=chat,
        )

    parse = {
        "mini-swe": mini_swe_format.parse_assistant,
        "open-swe": open_swe_format.parse_assistant,
        "qwen-tools": qwen_agent_chat.parse_assistant,
    }[agent_format]

    def local(conversations):
        replies, prompts, index = [None] * len(conversations), [], []
        for i, messages in enumerate(conversations):
            ids = tokenizer.encode(prompt_text(messages), add_special_tokens=False)
            if len(ids) + REFERENCE_MAX_NEW_TOKENS <= context:
                prompts.append({"prompt_token_ids": ids})
                index.append(i)
        if prompts:
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for i, output in zip(index, outputs, strict=True):
                replies[i] = parse(output.outputs[0].text)
        return replies

    header["agent_format"] = agent_format
    header["context"] = context
    header["chat_format"] = "qwen3, thinking off" if chat else "as trained"
    header["sampling"] = {"temperature": temperature, **(SAMPLING if temperature > 0 else {})}
    # Open-SWE's instructions name /testbed as the working directory.
    # Open-SWE's instructions work in /testbed: the repository moves there (find and
    # other tools do not descend into a symlinked starting directory) and /work/repo,
    # which the harness uses, links to it.
    setup_command = (
        f"cd / && mv /work/repo {open_swe_format.WORKDIR} && ln -s {open_swe_format.WORKDIR} /work/repo"
        if agent_format == "open-swe"
        else None
    )
    header["max_turns"] = max_turns or agent.MAX_TURNS
    return _evaluate_heldout(
        local, header, name, run, limit, workers, deadline, setup_command, max_turns
    )


@app.function(
    image=train_image, gpu="H100", cpu=8, memory=65536, timeout=1800, retries=0,
    volumes={"/hf": hf_cache},
)  # fmt: skip
def check_reference_family(deadline):
    """`trainer.reference` on one H100: the summed loss of two conversations packed in one
    window against plain Transformers SDPA on each conversation alone (same weights,
    bf16 autocast), and seconds per training step on a full 40,960-token window."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import torch
    from transformers import AutoModelForCausalLM

    from trainer.reference import ATTENTION, PackedLM, register_attention

    base, eos = "Qwen/Qwen3-0.6B-Base", 151643
    generator = torch.Generator().manual_seed(0)
    documents = [torch.randint(0, 150000, (n,), generator=generator) for n in (700, 500)]
    window = torch.cat([torch.cat([d, torch.tensor([eos])]) for d in documents])
    inputs = torch.full((1, 2048), eos)
    inputs[0, : len(window)] = window
    labels = torch.full_like(inputs, -100)
    labels[0, : len(window) - 1] = window[1:]
    labels[0, len(documents[0])] = -100  # the first EOS does not predict the next document
    plain = AutoModelForCausalLM.from_pretrained(
        base, revision=REFERENCE_MODELS[base], dtype=torch.float32, attn_implementation="sdpa"
    ).cuda()
    reference = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for document in documents:
            ids = torch.cat([document, torch.tensor([eos])]).cuda()[None]
            logits = plain(input_ids=ids).logits.float()[0, :-1]
            reference += float(torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum"))
    del plain
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3

    register_attention()
    apply_liger_kernel_to_qwen3(rope=True, rms_norm=True, swiglu=True, cross_entropy=False,
                                fused_linear_cross_entropy=False)  # fmt: skip
    lm = AutoModelForCausalLM.from_pretrained(
        base, revision=REFERENCE_MODELS[base], dtype=torch.float32, attn_implementation=ATTENTION
    ).cuda()
    lm.config.use_cache = False
    model = PackedLM(lm, eos).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        packed = float(model(inputs.cuda(), labels.cuda()))
    long = torch.randint(0, 150000, (1, 40960), generator=generator).cuda()
    times = []
    for _ in range(4):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(long, long).backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        model.zero_grad(set_to_none=True)
    seconds = sorted(times[1:])[1]
    return {
        "separate_loss": reference,
        "packed_loss": packed,
        "relative_difference": abs(packed - reference) / reference,
        "seconds_per_40960_step": round(seconds, 3),
        "tokens_per_second": round(40960 / seconds),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    }


@app.local_entrypoint()
def check_reference():
    """Check `trainer.reference` on one H100 (see `check_reference_family`)."""
    print(json.dumps(check_reference_family.remote(time.time() + 3600), indent=2))


TERMINAL_BENCH = "terminal-bench/terminal-bench-2-1"
TERMINAL_BENCH_SECONDS = 9 * 60 * 60


def _wait_healthy(server, end):
    import urllib.request

    while time.time() < end:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited with {server.returncode}")
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=5)
            return
        except OSError:
            time.sleep(10)
    raise TimeoutError("vLLM did not become healthy")


@app.function(
    image=terminal_bench_image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=TERMINAL_BENCH_SECONDS,
    retries=0,
    volumes={"/runs": runs_volume, "/hf": hf_cache},
)
def terminal_bench(model, attempts, concurrency, deadline):
    """Terminal-Bench 2.1 for a `train_reference` export (`sft:<run>[@<export>]`): vLLM's
    OpenAI server on localhost and Harbor's Terminus 2 in the same container, task
    sandboxes on Modal (each capped below this call's end), temperature 0.7. Harbor's
    job folder goes to `terminal-bench/<run>/` on the runs volume; returns the reward
    summary."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import subprocess

    started = time.time()
    end = min(deadline - 900, started + TERMINAL_BENCH_SECONDS - 900)
    folder, trained = _reference_export(model)
    name = model.removeprefix("sft:").replace("@", "-")
    server = subprocess.Popen(
        ["vllm", "serve", str(folder), "--served-model-name", name, "--port", "8000",
         "--max-model-len", str(trained["context"]), "--enable-prefix-caching", "--seed", "0"],
    )  # fmt: skip
    out = Path("/runs", "terminal-bench", name)
    out.mkdir(parents=True, exist_ok=True)
    try:
        _wait_healthy(server, end)
        model_info = {"max_input_tokens": trained["context"], "max_output_tokens": 4096,
                      "input_cost_per_token": 0, "output_cost_per_token": 0}  # fmt: skip
        command = [
            "harbor", "run", "-d", TERMINAL_BENCH, "-a", "terminus-2",
            "-m", f"hosted_vllm/{name}", "--ak", "api_base=http://localhost:8000/v1",
            "--ak", "temperature=0.7", "--ak", f"model_info={json.dumps(model_info)}",
            "-e", "modal", "--ek", f"sandbox_timeout_secs={int(end - time.time())}",
            "-k", str(attempts), "-n", str(concurrency), "-o", str(out),
        ]  # fmt: skip
        # Harbor's preflight looks for a Modal token file or variables; inside a Modal
        # container the SDK authenticates with the container's own credentials.
        Path.home().joinpath(".modal.toml").touch()
        env = {**os.environ, "HOSTED_VLLM_API_KEY": "none"}
        subprocess.run(command, env=env, timeout=max(60, end - time.time()), check=False)
    finally:
        server.terminate()
        runs_volume.commit()
    rewards, exceptions = [], {}
    for result in out.glob("*/*/result.json"):
        data = json.loads(result.read_text())
        reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        rewards.append(reward == 1)
        kind = (data.get("exception_info") or {}).get("exception_type") or "none"
        exceptions[kind] = exceptions.get(kind, 0) + 1
    summary = {"model": model, "dataset": TERMINAL_BENCH, "trials": len(rewards),
               "solved": sum(rewards), "exceptions": exceptions,
               "seconds": round(time.time() - started)}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    runs_volume.commit()
    return summary


def _spawn(function, *arguments):
    _spawn_all([(function, arguments)])


def _spawn_all(launches):
    """Spawn each (function, arguments) with one launch deadline and one supervisor."""
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    calls = [function.spawn(*arguments, deadline) for function, arguments in launches]
    ids = [c.object_id for c in calls]
    supervisor = modal_benchmarks.enforce_deadline.spawn(ids, deadline)
    print(json.dumps({"calls": ids, "supervisor": supervisor.object_id, "deadline_unix": deadline}))


@app.local_entrypoint()
def solve(
    name: str = "pr-tasks-pilot",
    limit: int = 0,
    workers: int = 32,
    attempts: int = 2,
    run: str = "",
):
    run = run or f"glm-5.3-{int(time.time())}{f'-limit{limit}' if limit else ''}"
    print(json.dumps({"results": f"lm-exploration-hybrid-data/{name}/agentic/{run}/"}))
    _spawn(solve_tasks, name, run, limit, workers, attempts)


@app.local_entrypoint()
def evaluate(
    export: str,
    agent_format: str = "tools",
    temperature: float = 0.0,
    max_turns: int = 0,
    name: str = "pr-tasks-pilot",
    limit: int = 0,
    workers: int = 48,
    run: str = "",
):
    suffix = (
        (f"-{agent_format}" if agent_format != "tools" else "")
        + (f"-t{temperature}" if temperature else "")
        + (f"-turns{max_turns}" if max_turns else "")
    )
    run = run or (
        f"eval-{export.split('/')[0]}{suffix}-{int(time.time())}{f'-limit{limit}' if limit else ''}"
    )
    print(json.dumps({"results": f"lm-exploration-hybrid-data/{name}/agentic/{run}/"}))
    _spawn(evaluate_export, export, agent_format, temperature, name, run, limit, workers, max_turns)


@app.local_entrypoint()
def reference_train(config: str, run: str, hours: float = 20.0):
    """Train (or continue) `run` with a `trainer.reference` run config; `hours` bounds
    this launch."""
    print(json.dumps({"results": f"lm-exploration-hybrid-runs/{REFERENCE_FOLDER}/{run}/"}))
    _spawn_all([(train_reference, (config, run, int(hours * 3600)))])


@app.local_entrypoint()
def reference_bench(model: str, attempts: int = 1, concurrency: int = 24):
    """Terminal-Bench 2.1 for `sft:<run>[@<export>]`."""
    _spawn_all([(terminal_bench, (model, attempts, concurrency))])


@app.local_entrypoint()
def reference_evaluate(
    models: str = "Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B",
    agent_format: str = "qwen-tools",
    temperature: float = 0.0,
    max_turns: int = 0,
    name: str = "pr-tasks-pilot",
    limit: int = 0,
    workers: int = 48,
):
    """`models`: pinned models and `sft:<run>` outputs, comma-separated, all in one format."""
    stamp, launches = int(time.time()), []
    sampled = f"-t{temperature:g}" if temperature > 0 else ""
    sampled += f"-turns{max_turns}" if max_turns else ""
    for model in models.split(","):
        slug = model.removeprefix("sft:").split("/")[-1].lower()
        run = f"eval-{slug}{sampled}-{stamp}{f'-limit{limit}' if limit else ''}"
        print(json.dumps({"results": f"lm-exploration-hybrid-data/{name}/agentic/{run}/"}))
        arguments = (model, agent_format, temperature, name, run, limit, workers, max_turns or None)
        launches.append((evaluate_reference, arguments))
    _spawn_all(launches)


@app.local_entrypoint()
def prepare(recipe: str = "data/recipes/agentic-sft-v1.json"):
    settings = json.loads(Path(recipe).read_text())
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    print(json.dumps(prepare_trajectories.remote(settings, deadline), indent=2))
