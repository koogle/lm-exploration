"""The RL training loop.

Each step draws `prompts_per_step` training problems of the pool, `math_share` of them
math, samples `samples_per_prompt` answers to each with the current weights and rewards
them (`rollouts.py`). The algorithm (`algorithms.py`) turns the rewards into one advantage
per answer; answers with zero advantage (every answer of a group scored the same) are
skipped. The rest are shuffled into the algorithm's `updates_per_batch` equal shares, one
optimizer update each: one update per step on-policy, several off-policy. For off-policy
updates, the log-probabilities of the weights that sampled the batch are computed once
before its first update.

Each answer's token loss is divided by a constant, the answers per update times the
task's token limit (no per-answer length normalization, as in Dr. GRPO,
arXiv:2503.20783). `kl_coef` adds the k3 estimate of KL(policy ‖ parent) on the answer
tokens, which is always measured and logged; `sampler_kl_coef` adds that of
KL(sampler ‖ learner). The sampler (cached one-token decoding) and the learner (a full
forward pass) compute probabilities with different kernels, so they differ slightly; the
mean sampler minus learner log-probability of the sampled tokens is logged as
`sampler_mismatch`. Pretraining replay (`replay_*`, the hook
`trainer/replay_hook.py` that SFT also uses) adds its loss at `replay_weight` to every
update. Gradients are clipped to norm 1.

Validation is greedy accuracy on the pool's validation problems, and replay validation
loss. The parent is an SFT checkpoint (`trainer/sft.py`) on the run volume. A run does not
resume; an interrupted run is rerun from scratch.
"""

import json
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

from training_data import rl_tasks
from training_data.subword import (
    BOS_TOKEN_ID,
    DEFAULT_FOLDER,
    EOS_TOKEN_ID,
    LOSS_IGNORE_INDEX,
    TOKENIZER_SHA256,
    digest,
    load_frozen,
)

from trainer import batches, rollout
from trainer.batches import collate
from trainer.replay_hook import ReplayHook
from trainer.rl import scores
from trainer.rl.algorithms import Tokens
from trainer.rl.rollouts import Scheduler, collect

FORMAT = "rl-v1"
# Parents this trainer accepts: SFT checkpoints of `trainer/sft.py`.
PARENT_FORMATS = ("math-sft-v1", "code-sft-v1", "mixture-sft-v1")
# Longest pool prompt used, in tokens.
MAX_PROMPT_TOKENS = 512


def token_logprobs(network, x, y):
    """Log-probabilities of the targets `y` (0 where `y` is ignored), and the target mask."""
    import torch
    from torch.nn import functional as F

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = network(x)
    mask = y != LOSS_IGNORE_INDEX
    values = -F.cross_entropy(
        logits.float().flatten(0, 1), y.clamp(min=0).flatten(), reduction="none"
    ).view_as(y)
    return values * mask, mask


def source_files():
    """{name: path} of the code a run depends on, hashed and copied into the run."""
    import model_10m.model as shared_module
    import model_100m.model as model_module
    import training_data.sandbox.execution as execution_module
    import training_data.sandbox.runner as runner_module

    package = Path(__file__).parent
    return {
        **{f"rl_{path.stem}": str(path) for path in sorted(package.glob("*.py"))},
        "rollout": rollout.__file__,
        "batches": batches.__file__,
        "rl_tasks": rl_tasks.__file__,
        "math_sft": sys.modules["training_data.math_sft"].__file__,
        "code_sft": sys.modules["training_data.code_sft"].__file__,
        "sandbox_execution": execution_module.__file__,
        "sandbox_runner": runner_module.__file__,
        **ReplayHook.SOURCES,
        "model": model_module.__file__,
        "shared_model": shared_module.__file__,
    }


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
    import torch
    from model_100m.model import GPT, ModelConfig

    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing RL work")
    if not torch.cuda.is_available():
        raise RuntimeError("Training is CUDA/Modal only")
    algorithm, rollouts = settings.algorithm, settings.rollouts
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
    files = source_files()
    source_hashes = {name: digest(path) for name, path in files.items()}
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
        model.parameters(), lr=settings.optimizer.lr, weight_decay=settings.optimizer.weight_decay
    )
    replay = ReplayHook.attach(settings, data_root, config["block_size"])
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
    shutil.copytree(DEFAULT_FOLDER, output / "tokenizer")
    for name, path in files.items():
        shutil.copyfile(path, output / f"{name}.py")
    scheduler = Scheduler(pool["train"], rollouts.rest_steps, rng)
    step, update = 0, 0

    def evaluate():
        metrics = {}
        batch = collect(model, pool["validation"], rollouts, 0, generator, tokenizer, sandbox)
        for task in rl_tasks.TASKS:
            picked = [i for i, row in enumerate(batch.rows) if row["task"] == task]
            if picked:
                metrics[f"validation_{task}_accuracy"] = sum(
                    batch.rewards[i] for i in picked
                ) / len(picked)
                metrics[f"validation_{task}_truncated"] = 1 - sum(
                    batch.finished[i] for i in picked
                ) / len(picked)
        if replay:
            model.eval()
            metrics["replay_validation_by_source"] = replay.validation(model)
            model.train()
        return metrics

    def old_logprobs(sequences, answers):
        """{answer: log-probabilities of its targets} under the current weights."""
        result = {}
        with torch.no_grad():
            for start in range(0, len(answers), settings.micro_batch):
                chosen = answers[start : start + settings.micro_batch]
                x, y = collate([sequences[i] for i in chosen], "cuda")
                logp, _ = token_logprobs(model, x, y)
                for b, i in enumerate(chosen):
                    result[i] = logp[b, : len(sequences[i]["y"])]
        return result

    def optimize(batch, sequences, advantage, answers, old, totals):
        """One optimizer update on `answers`; adds token statistics to `totals`."""
        nonlocal update
        for g in optimizer.param_groups:
            g["lr"] = settings.optimizer.learning_rate(update)
        optimizer.zero_grad(set_to_none=True)
        # Dr. GRPO's constant: the answers per update times the task's token limit.
        per_update = len(batch.rows) / algorithm.updates_per_batch
        for start in range(0, len(answers), settings.micro_batch):
            chosen = answers[start : start + settings.micro_batch]
            x, y = collate([sequences[i] for i in chosen], "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).float()
            mask = y != LOSS_IGNORE_INDEX
            logp = logits.log_softmax(-1).gather(-1, y.clamp(min=0)[..., None])[..., 0] * mask
            if old is None:
                old_logp = logp.detach()
            else:
                old_logp = torch.zeros_like(logp)
                for b, i in enumerate(chosen):
                    old_logp[b, : len(old[i])] = old[i]
            with torch.no_grad():
                reference_logp, _ = token_logprobs(reference, x, y)
            scale = torch.tensor(
                [1 / (per_update * rollouts.token_limit(batch.rows[i]["task"])) for i in chosen],
                device="cuda",
            )
            weight = torch.tensor([advantage[i] for i in chosen], device="cuda")
            # The sampler's records at each answer target position.
            top = algorithm.sampler_top_k
            sampled = torch.zeros(y.shape, device="cuda")
            top_ids = torch.zeros((*y.shape, top), dtype=torch.long, device="cuda")
            top_logprobs = torch.zeros((*y.shape, top), device="cuda")
            for b, i in enumerate(chosen):
                at = len(batch.rows[i]["prompt_ids"]) - 1
                logprobs, ids, values = batch.sampler[i]
                sampled[b, at : at + len(logprobs)] = logprobs
                if top:
                    top_ids[b, at : at + len(ids)] = ids
                    top_logprobs[b, at : at + len(ids)] = values
            tokens = Tokens(logits, y, mask, logp, old_logp.detach(), weight, scale)
            if top:
                tokens.sampler_top_ids, tokens.sampler_top_logprobs = top_ids, top_logprobs
            loss, stats = algorithm.loss(tokens)
            del logits, tokens
            delta = (reference_logp - logp) * mask
            kl = (delta.exp() - delta - 1) * mask
            if settings.kl_coef:
                loss = loss + settings.kl_coef * (scale[:, None] * kl).sum()
            if settings.sampler_kl_coef:
                drift = scores.sampler_kl(logp, sampled, mask)
                loss = loss + settings.sampler_kl_coef * (scale[:, None] * drift).sum()
            loss.backward()
            totals["mismatch"] += ((sampled - logp.detach()) * mask).sum().item()
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value
            totals["kl"] += kl.detach().sum().item()
            totals["nll"] += -logp.detach().sum().item()
            totals["tokens"] += int(mask.sum())
        if replay:
            totals["replay_loss"] += replay.backward(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Non-finite RL gradient")
        if answers or replay:
            optimizer.step()
        totals["grad_norm"] += grad_norm.item()
        update += 1

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

    def record(**values):
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": step, **values}) + "\n")

    started = time.monotonic()
    record(**evaluate(), phase="baseline")
    commit()
    print(json.dumps({"stage": "training", "target": settings.steps}), flush=True)
    last_saved = -1
    group = rollouts.samples_per_prompt
    while (
        step < settings.steps
        and time.monotonic() - started < settings.max_seconds
        and (deadline is None or time.time() < deadline)
    ):
        timings = {}
        clock = time.monotonic()
        math_prompts = round(rollouts.prompts_per_step * rollouts.math_share)
        drawn = scheduler.draw("math", math_prompts, step) + scheduler.draw(
            "code", rollouts.prompts_per_step - math_prompts, step
        )
        rows = [row for row in drawn for _ in range(group)]
        batch = collect(
            model,
            rows,
            rollouts,
            rollouts.temperature,
            generator,
            tokenizer,
            sandbox,
            algorithm.sampler_top_k,
        )
        timings["generate_seconds"] = time.monotonic() - clock
        clock = time.monotonic()
        advantage = algorithm.advantages(batch.rewards, group)
        for index, row in enumerate(drawn):
            mean = sum(batch.rewards[index * group : (index + 1) * group]) / group
            scheduler.update(row, mean, step)
        sequences = batch.sequences()
        trained = [i for i, a in enumerate(advantage) if a != 0]
        rng.shuffle(trained)
        old = old_logprobs(sequences, trained) if algorithm.policy == "off" else None
        totals = {
            "kl": 0.0,
            "nll": 0.0,
            "mismatch": 0.0,
            "tokens": 0,
            "grad_norm": 0.0,
            "replay_loss": 0.0,
        }
        shares = algorithm.updates_per_batch
        for share in range(shares):
            answers = trained[share * len(trained) // shares : (share + 1) * len(trained) // shares]
            optimize(batch, sequences, advantage, sorted(answers), old, totals)
        step += 1
        timings["train_seconds"] = time.monotonic() - clock
        values = {}
        for task in rl_tasks.TASKS:
            picked = [i for i, row in enumerate(rows) if row["task"] == task]
            if not picked:
                continue
            groups = [picked[k : k + group] for k in range(0, len(picked), group)]
            values[f"{task}_reward"] = sum(batch.rewards[i] for i in picked) / len(picked)
            values[f"{task}_uniform_groups"] = sum(
                len({batch.rewards[i] for i in g}) == 1 for g in groups
            ) / len(groups)
            values[f"{task}_answer_tokens"] = sum(len(batch.completions[i]) for i in picked) / len(
                picked
            )
            values[f"{task}_truncated"] = 1 - sum(batch.finished[i] for i in picked) / len(picked)
        tokens = max(1, totals.pop("tokens"))
        values.update(
            trained_answers=len(trained),
            sampled_nll=totals.pop("nll") / tokens,
            kl=totals.pop("kl") / tokens,
            sampler_mismatch=totals.pop("mismatch") / tokens,
            grad_norm=totals.pop("grad_norm") / shares,
            lr=settings.optimizer.learning_rate(update - 1),
            updates=update,
            resting_prompts=scheduler.resting(step),
            **({"replay_loss": totals.pop("replay_loss") / shares} if replay else {}),
            **{key: value / tokens for key, value in totals.items() if key != "replay_loss"},
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
