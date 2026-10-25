"""Bounded single-GPU RL on math and code problems with checkable answers.

What is trained: the policy writes its own answers and is rewarded for correct ones. Each
step draws `prompts_per_step` training problems of the pool (`training_data.rl_tasks`),
`math_share` of them math, and samples `samples_per_prompt` complete answers to each with
the current weights at `temperature` (`trainer/rollout.py`). An answer scores 1 if it
ends in EOS and is correct (Math-Verify for math, every unit test passing in the code
sandbox for code), else 0.

The advantage of an answer is its reward minus the mean reward of its prompt's group.
The loss is the policy gradient −advantage × log p(token) summed over every answer token
and divided by a constant: the number of answers times the task's token limit (no
per-group standard deviation or per-answer length normalization, as in Dr. GRPO,
arXiv:2503.20783). One optimizer step per batch, so samples always come from the current
weights and no importance ratio is needed. A group whose answers all score the same has
zero advantage and is skipped. A prompt whose group is all wrong or all right on two
visits in a row rests for `rest_steps` steps.

Optional terms: `kl_coef` times the k3 estimate of KL(policy ‖ parent) on the answer
tokens (always measured and logged), and the SFT trainer's pretraining replay
(`replay_*`, `training_data.replay`) at `replay_weight`.

Validation is greedy (temperature 0) accuracy on the pool's validation problems, and
replay validation loss. The parent is an SFT checkpoint (`trainer/sft.py`) on the run
volume. A run does not resume; an interrupted run is rerun from scratch.
"""

import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from training_data import rl_tasks
from training_data.subword import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    LOSS_IGNORE_INDEX,
    TOKENIZER_SHA256,
    digest,
    load_frozen,
)

FORMAT = "rl-v1"
# Parents this trainer accepts: SFT checkpoints of `trainer/sft.py`.
PARENT_FORMATS = ("math-sft-v1", "code-sft-v1", "mixture-sft-v1")
# Longest pool prompt used, in tokens.
MAX_PROMPT_TOKENS = 512


@dataclass(frozen=True)
class Settings:
    steps: int = 300
    prompts_per_step: int = 64
    samples_per_prompt: int = 8
    math_share: float = 0.5
    math_tokens: int = 256
    code_tokens: int = 384
    temperature: float = 1.0
    lr: float = 1e-6
    warmup_steps: int = 20
    weight_decay: float = 0.0
    micro_batch: int = 32
    kl_coef: float = 0.0
    rest_steps: int = 50
    seed: int = 42
    evaluate_every: int = 50
    save_every: int = 100
    max_seconds: int = 72000
    replay_snapshot: str = ""
    replay_sources: dict = field(default_factory=dict)
    replay_batch_size: int = 0
    replay_context: int = 1024
    replay_weight: float = 0.0
    replay_validation_windows: int = 4

    def __post_init__(self):
        for key in (
            "steps",
            "prompts_per_step",
            "samples_per_prompt",
            "math_tokens",
            "code_tokens",
            "micro_batch",
            "evaluate_every",
            "save_every",
            "max_seconds",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.samples_per_prompt < 2:
            raise ValueError("Group advantages need at least two samples per prompt")
        if not 0 <= self.math_share <= 1:
            raise ValueError("math_share must be in [0, 1]")
        if not math.isfinite(self.lr) or not 0 < self.lr <= 1e-4:
            raise ValueError("Invalid learning rate")
        if not 0 <= self.warmup_steps < self.steps or not 0 <= self.rest_steps:
            raise ValueError("Invalid warmup or rest steps")
        if not 0 < self.temperature <= 2 or self.kl_coef < 0:
            raise ValueError("Invalid temperature or KL coefficient")
        if self.weight_decay < 0:
            raise ValueError("Invalid weight_decay")
        if self.max_seconds > 20 * 60 * 60:
            raise ValueError("RL capped at 20 hours per invocation")
        replay = (self.replay_snapshot, self.replay_sources, self.replay_batch_size)
        if (any(replay) or self.replay_weight) and not (all(replay) and 0 < self.replay_weight < 1):
            raise ValueError("Replay needs a snapshot, sources, batch size and weight")

    def learning_rate(self, step):
        """Linear warmup over `warmup_steps`, then constant."""
        return self.lr * min(1.0, (step + 1) / max(1, self.warmup_steps))

    def token_limit(self, task):
        return self.math_tokens if task == "math" else self.code_tokens


class Scheduler:
    """Draws training prompts per task; a prompt solved by none or all of its group on
    two visits in a row rests for `rest` steps."""

    def __init__(self, rows, rest, rng):
        self.rows, self.rest, self.rng = rows, rest, rng
        self.state = {}

    def draw(self, task, count, step):
        rows = [r for r in self.rows if r["task"] == task]
        ready = [r for r in rows if self.state.get(r["id"], {}).get("until", 0) <= step]
        return self.rng.sample(ready if len(ready) >= count else rows, count)

    def update(self, row, mean, step):
        state = self.state.setdefault(row["id"], {"until": 0, "extreme": None, "streak": 0})
        if mean in (0.0, 1.0):
            state["streak"] = state["streak"] + 1 if state["extreme"] == mean else 1
            state["extreme"] = mean
            if state["streak"] >= 2:
                state.update(until=step + self.rest, streak=0, extreme=None)
        else:
            state.update(streak=0, extreme=None)

    def resting(self, step):
        return sum(s["until"] > step for s in self.state.values())


def advantages(rewards, group):
    """Reward minus the mean of its group of `group` consecutive answers."""
    result = []
    for start in range(0, len(rewards), group):
        chunk = rewards[start : start + group]
        mean = sum(chunk) / len(chunk)
        result += [r - mean for r in chunk]
    return result


def train(
    parent,
    pool_path,
    output,
    settings,
    sandbox,
    commit=lambda: None,
    deadline=None,
    data_root=Path("/data"),
):
    """Train until the step target, `settings.max_seconds` or the absolute `deadline`;
    then validate and save a final checkpoint. `sandbox.run` checks code answers."""
    import shutil

    import torch
    from model_100m.model import GPT, ModelConfig
    from torch.nn import functional as F

    from trainer import rollout

    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing RL work")
    if not torch.cuda.is_available():
        raise RuntimeError("Training is CUDA/Modal only")
    tokenizer = load_frozen()
    pool, pool_hash = rl_tasks.load(pool_path)
    for split in pool:
        for row in pool[split]:
            row["prompt_ids"] = [BOS_TOKEN_ID] + tokenizer.encode(
                rl_tasks.prompt(row), add_special_tokens=False
            ).ids
        pool[split] = [r for r in pool[split] if len(r["prompt_ids"]) <= MAX_PROMPT_TOKENS]
    # Only load our trusted private run-volume checkpoints, never dataset-supplied files.
    saved = torch.load(parent, map_location="cpu", weights_only=False)
    if saved.get("format") not in PARENT_FORMATS:
        raise ValueError("Expected an SFT parent checkpoint")
    import model_10m.model as shared_module
    import model_100m.model as model_module
    import training_data.code_execution as execution_module
    import training_data.replay as replay_module

    source_hashes = {
        "trainer": digest(__file__),
        "rollout": digest(rollout.__file__),
        "rl_tasks": digest(rl_tasks.__file__),
        "math_sft": digest(sys.modules["training_data.math_sft"].__file__),
        "code_sft": digest(sys.modules["training_data.code_sft"].__file__),
        "code_execution": digest(execution_module.__file__),
        "replay": digest(replay_module.__file__),
        "model": digest(model_module.__file__),
        "shared_model": digest(shared_module.__file__),
    }
    if saved["identity"]["source_sha256"]["model"] != source_hashes["model"]:
        raise ValueError("Model implementation differs from parent checkpoint")
    if saved["identity"]["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise ValueError("Parent uses another tokenizer")
    config = saved["model_config"]
    model = GPT(ModelConfig(**config)).cuda()
    model.load_state_dict(saved["model"], strict=True)
    reference = GPT(ModelConfig(**config)).cuda().eval()
    reference.load_state_dict(saved["model"], strict=True)
    reference.requires_grad_(False)
    del saved
    torch.manual_seed(settings.seed)
    rng = random.Random(settings.seed)
    generator = torch.Generator(device="cuda").manual_seed(settings.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay
    )
    replay, replay_validation = None, []
    if settings.replay_batch_size:
        if settings.replay_context > config["block_size"]:
            raise ValueError("Replay context exceeds the model's context")
        replay = replay_module.Replay(
            Path(data_root) / settings.replay_snapshot,
            settings.replay_sources,
            settings.replay_context,
            tokenizer,
            settings.seed,
        )
        replay_validation = replay.validation(settings.replay_validation_windows)
    identity = {
        "parent": str(parent),
        "parent_sha256": digest(parent),
        "pool_sha256": pool_hash,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "replay": replay.identity() if replay else None,
        "settings": asdict(settings),
        "source_sha256": source_hashes,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(identity, indent=2))
    shutil.copyfile(pool_path, output / "pool.jsonl")
    from training_data.subword import DEFAULT_FOLDER

    shutil.copytree(DEFAULT_FOLDER, output / "tokenizer")
    files = {
        "trainer": __file__,
        "rollout": rollout.__file__,
        "rl_tasks": rl_tasks.__file__,
        "math_sft": sys.modules["training_data.math_sft"].__file__,
        "code_sft": sys.modules["training_data.code_sft"].__file__,
        "code_execution": execution_module.__file__,
        "replay": replay_module.__file__,
        "model": model_module.__file__,
        "shared_model": shared_module.__file__,
    }
    for name, path in files.items():
        shutil.copyfile(path, output / f"{name}.py")
    scheduler = Scheduler(pool["train"], settings.rest_steps, rng)
    step = 0

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    def generate(rows, temperature):
        """(completions, finished, rewards) for `rows`, one batch per task; code answers
        are checked in a background thread while math is sampled."""
        completions, finished = [None] * len(rows), [None] * len(rows)
        rewards = [0.0] * len(rows)
        model.eval()
        with ThreadPoolExecutor(max_workers=1) as pool_thread:
            pending = None
            for task in ("code", "math"):
                indices = [i for i, r in enumerate(rows) if r["task"] == task]
                if not indices:
                    continue
                samples = rollout.sample(
                    model,
                    [rows[i]["prompt_ids"] for i in indices],
                    settings.token_limit(task),
                    EOS_TOKEN_ID,
                    temperature,
                    generator,
                )
                ids, done = samples.completions, samples.finished
                for i, c, d in zip(indices, ids, done, strict=True):
                    completions[i], finished[i] = c, d
                chosen = [rows[i] for i in indices]
                texts = [decode(c) for c in ids]
                if task == "code":
                    pending = (
                        indices,
                        pool_thread.submit(rl_tasks.rewards, chosen, texts, done, sandbox),
                    )
                else:
                    for i, r in zip(
                        indices, rl_tasks.rewards(chosen, texts, done, sandbox), strict=True
                    ):
                        rewards[i] = r
            if pending:
                for i, r in zip(pending[0], pending[1].result(), strict=True):
                    rewards[i] = r
        model.train()
        return completions, finished, rewards

    def sequences(rows, completions, finished):
        """Inputs and targets of prompt + answer (+ EOS); only answer tokens are targets."""
        result = []
        for row, completion, done in zip(rows, completions, finished, strict=True):
            answer = completion + ([EOS_TOKEN_ID] if done else [])
            tokens = row["prompt_ids"] + answer
            labels = [LOSS_IGNORE_INDEX] * len(row["prompt_ids"]) + answer
            result.append((tokens[:-1], labels[1:]))
        return result

    def collate(batch):
        width = max(len(x) for x, _ in batch)
        x = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
        y = torch.full_like(x, LOSS_IGNORE_INDEX)
        for i, (tokens, labels) in enumerate(batch):
            x[i, : len(tokens)] = torch.tensor(tokens, device="cuda")
            y[i, : len(labels)] = torch.tensor(labels, device="cuda")
        return x, y

    def token_logprobs(network, x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = network(x)
        mask = y != LOSS_IGNORE_INDEX
        values = -F.cross_entropy(
            logits.float().flatten(0, 1),
            y.clamp(min=0).flatten(),
            reduction="none",
        ).view_as(y)
        return values * mask, mask

    def replay_loss(rows):
        x, y = collate([(r["x"], r["y"]) for r in rows])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        value = F.cross_entropy(
            logits.float().flatten(0, 1), y.flatten(), ignore_index=LOSS_IGNORE_INDEX
        )
        return value

    def evaluate():
        metrics = {}
        rows = pool["validation"]
        _, finished, rewards = generate(rows, 0)
        for task in rl_tasks.TASKS:
            chosen = [r for r, row in zip(rewards, rows, strict=True) if row["task"] == task]
            done = [f for f, row in zip(finished, rows, strict=True) if row["task"] == task]
            if chosen:
                metrics[f"validation_{task}_accuracy"] = sum(chosen) / len(chosen)
                metrics[f"validation_{task}_truncated"] = 1 - sum(done) / len(done)
        if replay_validation:
            model.eval()
            totals = {}
            with torch.inference_mode():
                for row in replay_validation:
                    totals.setdefault(row["bucket"], []).append(replay_loss([row]).item())
            model.train()
            metrics["replay_validation_by_source"] = {
                name: sum(v) / len(v) for name, v in sorted(totals.items())
            }
        return metrics

    def record(**values):
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": step, **values}) + "\n")

    def save():
        target = output / f"step-{step:09d}.pt"
        temporary = target.with_suffix(".partial")
        torch.save(
            {
                "format": FORMAT,
                "identity": identity,
                "special_tokens": {"bos": BOS_TOKEN_ID, "eos": EOS_TOKEN_ID},
                "model_config": asdict(model.config),
                "model": model.state_dict(),
                "step": step,
                "prompt_state": scheduler.state,
            },
            temporary,
        )
        temporary.replace(target)
        commit()
        return str(target)

    started = time.monotonic()
    record(**evaluate(), phase="baseline")
    commit()
    print(json.dumps({"stage": "training", "target": settings.steps}), flush=True)
    last_saved = -1
    while (
        step < settings.steps
        and time.monotonic() - started < settings.max_seconds
        and (deadline is None or time.time() < deadline)
    ):
        timings = {}
        clock = time.monotonic()
        math_prompts = round(settings.prompts_per_step * settings.math_share)
        drawn = scheduler.draw("math", math_prompts, step) + scheduler.draw(
            "code", settings.prompts_per_step - math_prompts, step
        )
        group = settings.samples_per_prompt
        rows = [row for row in drawn for _ in range(group)]
        completions, finished, rewards = generate(rows, settings.temperature)
        timings["generate_seconds"] = time.monotonic() - clock
        clock = time.monotonic()
        advantage = advantages(rewards, group)
        for index, row in enumerate(drawn):
            scheduler.update(row, sum(rewards[index * group : (index + 1) * group]) / group, step)
        for g in optimizer.param_groups:
            g["lr"] = settings.learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        data = sequences(rows, completions, finished)
        trained = [i for i, a in enumerate(advantage) if a != 0]
        totals = {"kl": 0.0, "nll": 0.0, "tokens": 0}
        for start in range(0, len(trained), settings.micro_batch):
            chosen = trained[start : start + settings.micro_batch]
            x, y = collate([data[i] for i in chosen])
            logp, mask = token_logprobs(model, x, y)
            with torch.no_grad():
                reference_logp, _ = token_logprobs(reference, x, y)
            weight = torch.tensor(
                [
                    advantage[i] / (len(rows) * settings.token_limit(rows[i]["task"]))
                    for i in chosen
                ],
                device="cuda",
            )
            scale = torch.tensor(
                [1 / (len(rows) * settings.token_limit(rows[i]["task"])) for i in chosen],
                device="cuda",
            )
            loss = -(weight[:, None] * logp).sum()
            delta = (reference_logp - logp) * mask
            kl = (delta.exp() - delta - 1) * mask
            if settings.kl_coef:
                loss = loss + settings.kl_coef * (scale[:, None] * kl).sum()
            loss.backward()
            totals["kl"] += kl.detach().sum().item()
            totals["nll"] += -logp.detach().sum().item()
            totals["tokens"] += int(mask.sum())
        values = {}
        if replay:
            value = replay_loss([replay.sample() for _ in range(settings.replay_batch_size)])
            (settings.replay_weight * value).backward()
            values["replay_loss"] = value.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Non-finite RL gradient")
        if trained or replay:
            optimizer.step()
        step += 1
        timings["train_seconds"] = time.monotonic() - clock
        for task in rl_tasks.TASKS:
            picked = [i for i, row in enumerate(rows) if row["task"] == task]
            if not picked:
                continue
            groups = [picked[k : k + group] for k in range(0, len(picked), group)]
            values[f"{task}_reward"] = sum(rewards[i] for i in picked) / len(picked)
            values[f"{task}_uniform_groups"] = sum(
                len({rewards[i] for i in g}) == 1 for g in groups
            ) / len(groups)
            values[f"{task}_answer_tokens"] = sum(len(completions[i]) for i in picked) / len(picked)
            values[f"{task}_truncated"] = 1 - sum(finished[i] for i in picked) / len(picked)
        values.update(
            trained_answers=len(trained),
            sampled_nll=totals["nll"] / max(1, totals["tokens"]),
            kl=totals["kl"] / max(1, totals["tokens"]),
            grad_norm=grad_norm.item(),
            lr=settings.learning_rate(step - 1),
            resting_prompts=scheduler.resting(step),
            **timings,
        )
        record(**values)
        print(json.dumps({"step": step, **values}), flush=True)
        if step % settings.evaluate_every == 0:
            metrics = evaluate()
            record(**metrics)
            print(json.dumps({"step": step, **metrics}), flush=True)
        if step % settings.save_every == 0:
            save()
            last_saved = step
    record(**evaluate(), phase="final")
    path = str(output / f"step-{step:09d}.pt") if last_saved == step else save()
    commit()
    return {"checkpoint": path, "step": step, "seconds": time.monotonic() - started}
