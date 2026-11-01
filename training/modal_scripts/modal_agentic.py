"""Agent episodes on verified PR tasks (see `data/pr_task_agent.py`): teacher trajectories and evaluation.

Run from training/:
    modal run --detach -m modal_scripts.modal_agentic::solve [--limit N] [--workers 32]
    modal run --detach -m modal_scripts.modal_agentic::evaluate --export <run>/<export> [--limit N]
    modal run -m modal_scripts.modal_agentic::prepare [--recipe data/recipes/agentic-sft-v1.json]
    modal run --detach -m modal_scripts.modal_agentic::reference_train [--bases B1,B2] [--tag 01]
    modal run --detach -m modal_scripts.modal_agentic::reference_evaluate --models M1,sft:<run>

Tasks are the pilot's verified tasks with a clean issue (`pr-tasks-pilot/verified.jsonl`
and `issues.jsonl` on the `lm-exploration-hybrid-data` volume). Each episode installs
the repository in a sandbox with network (`modal_pr_tasks.installed_snapshot`), then
works in a sandbox from that snapshot without network, and is graded in a third one.

`solve` has GLM-5.3 (through OpenRouter, with the launching shell's
`OPENROUTER_API_KEY`) work every task with the tools (training repositories'
tasks `attempts` times, held-out ones once) and writes
`<name>/agentic/<run>/trajectories.jsonl` (messages, patch, grade, resolved, token
usage, held-out flag) and `summary.json`. `evaluate` runs a hybrid export (greedy,
GLM chat format) on the held-out repositories' tasks in one H100 call and writes
`<name>/agentic/<run>/` the same way. `prepare` turns the recipe's `trajectories` run
into the chat SFT subset `agentic` (resolved episodes from training repositories
only, rendered with `chat_sft.prepare_subset`).

The reference stages run Qwen3 models through the same tasks and data, to separate
the effect of broad pretraining from the agent data. `reference_train` fully
fine-tunes pinned Qwen3 base models on the recipe's resolved training-repository
trajectories (in Qwen3's chat format, `training_data.qwen_agent_chat`) and writes
`lm-exploration-hybrid-runs/agentic-reference/<run>/` (bf16 model, metrics, summary).
`reference_evaluate` runs pinned Qwen3 chat models (zero-shot, thinking off) or
those fine-tunes (`sft:<run>`) through vLLM on the held-out tasks, with the hybrid
evaluation's greedy decoding, turn and token limits, and writes the same files as
`evaluate`. Every call has `retries=0`, a timeout of at
most 6 hours and refuses work 24 hours after launch; sandboxes are capped at one
hour and at the deadline, and a supervisor cancels the spawned call at the deadline.
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
EVAL_SANDBOX_SECONDS = 5 * 60 * 60

# Reference models: Qwen3 with its broad pretraining, through the same tasks and data.
REFERENCE_MODELS = {
    "Qwen/Qwen3-0.6B-Base": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    "Qwen/Qwen3-1.7B-Base": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
REFERENCE_FOLDER = "agentic-reference"  # on the runs volume
REFERENCE_SFT = {
    "context": 32768,
    "epochs": 4,
    "accumulation": 8,
    "learning_rate": 2e-5,
    "warmup_steps": 10,
    "min_learning_rate_fraction": 0.1,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "loss_chunk": 4096,
    "seed": 0,
}
# As the hybrid evaluation (agent_policy.LocalPolicy): greedy, 1,536 new tokens per turn.
REFERENCE_MAX_NEW_TOKENS = 1536
# vLLM's FlashInfer kernels compile at load time, so the image carries nvcc (as in
# modal_teacher.py); training uses the torch and transformers that vLLM installs.
reference_image = add_project(
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.31.0", "huggingface-hub[hf_transfer]")
    .env(
        {
            "HF_HOME": "/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/project",
            "CUDA_HOME": "/usr/local/cuda",
        }
    )
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


def _summary(rows, started):
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
        "max_turns": agent.MAX_TURNS,
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
    timeout=6 * 60 * 60,
    retries=0,
    volumes={"/data": volume, "/runs": runs_volume},
)
def evaluate_export(export, name, run, limit, workers, deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import torch
    from model_hybrid import agent_policy, generate

    from tokenizer import glm

    run_name, export_name = export.split("/")
    lm = generate.load_export(
        Path("/runs", run_name, "export", export_name), dtype=torch.bfloat16, device="cuda"
    )
    local = agent_policy.LocalPolicy(lm, glm.load(), rows=96)
    return _evaluate_heldout(local, {"export": export}, name, run, limit, workers, deadline)


def _evaluate_heldout(local, header, name, run, limit, workers, deadline):
    """Run `local` (a batch policy) as the agent on the held-out repositories' tasks."""
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
            return Workspace(task, snapshot, deadline, record, EVAL_SANDBOX_SECONDS), None
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
        lambda episode, name, arguments: episode["workspace"].call(name, arguments),
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
    summary = {**header, "run": run, **_summary(rows, started)}
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
    image=reference_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=4 * 60 * 60,
    retries=0,
    volumes={"/data": volume, "/runs": runs_volume, "/hf": hf_cache},
)
def train_reference(base, run, trajectories, deadline):
    """Full fine-tune of a Qwen reference base on the resolved training-repository
    trajectories, in Qwen3's chat format; loss on the assistant turns only."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import math
    import random

    import torch
    from torch.utils.checkpoint import checkpoint
    from training_data import pr_task_agent as agent
    from training_data import qwen_agent_chat
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings, revision = REFERENCE_SFT, REFERENCE_MODELS[base]
    out = Path("/runs", REFERENCE_FOLDER, run)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
    rows = [json.loads(line) for line in Path("/data", trajectories).read_text().splitlines()]
    kept = [r for r in rows if r["resolved"] and not r["heldout"]]
    examples = []
    for row in kept:
        conversation = agent.training_row(row["messages"])["messages"]
        ids, mask = qwen_agent_chat.encode(
            lambda text: tokenizer.encode(text, add_special_tokens=False), conversation, agent.TOOLS
        )
        if len(ids) <= settings["context"]:
            examples.append((ids, mask))
    model = AutoModelForCausalLM.from_pretrained(
        base, revision=revision, dtype=torch.float32, attn_implementation="sdpa"
    ).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    head = model.get_output_embeddings()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=settings["weight_decay"],
        fused=True,
    )
    rng = random.Random(settings["seed"])
    order = []
    for _ in range(settings["epochs"]):
        epoch = list(range(len(examples)))
        rng.shuffle(epoch)
        order += epoch
    accumulation = settings["accumulation"]
    steps = math.ceil(len(order) / accumulation)

    def learning_rate(step):
        peak, floor = settings["learning_rate"], settings["min_learning_rate_fraction"]
        if step < settings["warmup_steps"]:
            return peak * (step + 1) / settings["warmup_steps"]
        progress = (step - settings["warmup_steps"]) / max(1, steps - settings["warmup_steps"])
        return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))

    def chunk_loss(hidden, targets, weights):
        losses = torch.nn.functional.cross_entropy(head(hidden).float(), targets, reduction="none")
        return (losses * weights).sum()

    started, metrics, stopped = time.time(), [], None
    for step in range(steps):
        if time.time() >= deadline - 1800:
            stopped = "launch deadline"
            break
        group = order[step * accumulation : (step + 1) * accumulation]
        target_tokens = sum(sum(examples[i][1][1:]) for i in group)
        loss_sum = 0.0
        for index in group:
            ids, mask = examples[index]
            inputs = torch.tensor(ids[:-1], device="cuda")[None]
            targets = torch.tensor(ids[1:], device="cuda")
            weights = torch.tensor(mask[1:], device="cuda", dtype=torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(input_ids=inputs).last_hidden_state[0]
                total = hidden.new_zeros((), dtype=torch.float32)
                for start in range(0, hidden.shape[0], settings["loss_chunk"]):
                    end = start + settings["loss_chunk"]
                    if weights[start:end].any():
                        total = total + checkpoint(
                            chunk_loss,
                            hidden[start:end],
                            targets[start:end],
                            weights[start:end],
                            use_reentrant=False,
                        )
            (total / target_tokens).backward()
            loss_sum += total.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step + 1,
            "loss": round(loss_sum / target_tokens, 5),
            "grad_norm": round(float(norm), 4),
            "learning_rate": learning_rate(step),
            "tokens": sum(len(examples[i][0]) for i in group),
            "target_tokens": target_tokens,
            "seconds": round(time.time() - started, 1),
        }
        metrics.append(record)
        print(json.dumps(record), flush=True)
    model.to(torch.bfloat16).save_pretrained(out / "model")
    tokenizer.save_pretrained(out / "model")
    with open(out / "metrics.jsonl", "w") as handle:
        handle.writelines(json.dumps(m) + "\n" for m in metrics)
    summary = {
        "base": base,
        "revision": revision,
        "trajectories": trajectories,
        "episodes": {"all": len(rows), "resolved_training_repositories": len(kept)},
        "examples": len(examples),
        "tokens_per_epoch": sum(len(ids) for ids, _ in examples),
        "target_tokens_per_epoch": sum(sum(mask) for _, mask in examples),
        "steps": len(metrics),
        "planned_steps": steps,
        "stopped": stopped,
        "settings": settings,
        "final_loss": metrics[-1]["loss"] if metrics else None,
        "seconds": round(time.time() - started),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    runs_volume.commit()
    return summary


@app.function(
    image=reference_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=6 * 60 * 60,
    retries=0,
    volumes={"/data": volume, "/runs": runs_volume, "/hf": hf_cache},
)
def evaluate_reference(model, name, run, limit, workers, deadline):
    """A Qwen reference model as the agent on the held-out tasks, through vLLM.

    `model` is a pinned Qwen3 chat model (zero-shot, thinking off) or `sft:<run>`, a
    `train_reference` output (rendered without the empty thinking block, as trained).
    """
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    from training_data import pr_task_agent as agent
    from training_data import qwen_agent_chat
    from vllm import LLM, SamplingParams

    context = REFERENCE_SFT["context"]
    if model.startswith("sft:"):
        runs_volume.reload()
        folder = Path("/runs", REFERENCE_FOLDER, model.removeprefix("sft:"))
        trained = json.loads((folder / "summary.json").read_text())
        source, revision, chat = str(folder / "model"), None, False
        header = {"model": model, "base": trained["base"], "base_revision": trained["revision"]}
    else:
        source, revision, chat = model, REFERENCE_MODELS[model], True
        header = {"model": model, "revision": revision}
    llm = LLM(
        model=source,
        revision=revision,
        max_model_len=context,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        temperature=0.0,
        max_tokens=REFERENCE_MAX_NEW_TOKENS,
        stop_token_ids=[tokenizer.convert_tokens_to_ids(qwen_agent_chat.TURN_END)],
        skip_special_tokens=False,
    )

    def local(conversations):
        replies, prompts, index = [None] * len(conversations), [], []
        for i, messages in enumerate(conversations):
            text = qwen_agent_chat.render(
                agent.training_row(messages)["messages"],
                agent.TOOLS,
                add_generation_prompt=True,
                chat=chat,
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) + REFERENCE_MAX_NEW_TOKENS <= context:
                prompts.append({"prompt_token_ids": ids})
                index.append(i)
        if prompts:
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for i, output in zip(index, outputs, strict=True):
                replies[i] = qwen_agent_chat.parse_assistant(output.outputs[0].text)
        return replies

    header["chat_format"] = "qwen3, thinking off" if chat else "qwen3, as trained"
    return _evaluate_heldout(local, header, name, run, limit, workers, deadline)


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
    export: str, name: str = "pr-tasks-pilot", limit: int = 0, workers: int = 48, run: str = ""
):
    run = (
        run or f"eval-{export.split('/')[0]}-{int(time.time())}{f'-limit{limit}' if limit else ''}"
    )
    print(json.dumps({"results": f"lm-exploration-hybrid-data/{name}/agentic/{run}/"}))
    _spawn(evaluate_export, export, name, run, limit, workers)


@app.local_entrypoint()
def reference_train(
    bases: str = "Qwen/Qwen3-0.6B-Base,Qwen/Qwen3-1.7B-Base",
    recipe: str = "data/recipes/agentic-sft-v1.json",
    tag: str = "01",
):
    trajectories = json.loads(Path(recipe).read_text())["trajectories"]
    runs = {b: f"{b.split('/')[1].lower()}-agentic-sft-{tag}" for b in bases.split(",")}
    print(json.dumps({"results": f"lm-exploration-hybrid-runs/{REFERENCE_FOLDER}/", **runs}))
    _spawn_all([(train_reference, (b, r, trajectories)) for b, r in runs.items()])


@app.local_entrypoint()
def reference_evaluate(
    models: str = "Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B",
    name: str = "pr-tasks-pilot",
    limit: int = 0,
    workers: int = 48,
):
    """`models`: pinned Qwen3 chat models and `sft:<run>` outputs, comma-separated."""
    stamp, launches = int(time.time()), []
    for model in models.split(","):
        slug = model.removeprefix("sft:").split("/")[-1].lower()
        run = f"eval-{slug}-{stamp}{f'-limit{limit}' if limit else ''}"
        print(json.dumps({"results": f"lm-exploration-hybrid-data/{name}/agentic/{run}/"}))
        launches.append((evaluate_reference, (model, name, run, limit, workers)))
    _spawn_all(launches)


@app.local_entrypoint()
def prepare(recipe: str = "data/recipes/agentic-sft-v1.json"):
    settings = json.loads(Path(recipe).read_text())
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    print(json.dumps(prepare_trajectories.remote(settings, deadline), indent=2))
