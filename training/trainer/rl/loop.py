"""The RL training loop.

Each step takes one `Batch` of answers with rewards from the trajectory source
(`settings.trajectories`): sampled by the current weights (`rollouts.Sampler`), sampled
and reused for several updates, or read from a trajectory file (`off_policy/sources.py`).
The algorithm (`algorithms.py`) gives one advantage per answer; policy-gradient
algorithms skip answers with zero advantage (every answer of a group scored the same).
The answers are shuffled into the source's `updates_per_batch` equal shares, one
optimizer update each. When the algorithm needs the sampling weights' log-probabilities,
they are computed once before the batch's first update.

Each answer's token loss is divided by a constant, the answers per update times the
task's token limit (no per-answer length normalization, as in Dr. GRPO,
arXiv:2503.20783). `kl_coef` adds the k3 estimate of KL(policy ‖ parent) on the answer
tokens, which is always measured and logged. With `stable_reinforce`, the hook of
`off_policy/stable_reinforce.py` corrects every micro-batch for the difference between
sampler and learner probabilities. Pretraining replay (`replay_*`, the hook
`trainer/replay_hook.py` that SFT also uses) adds its loss at `replay_weight` to every
update. Gradients are clipped to norm 1.

Validation is greedy accuracy on the pool's validation problems, and replay validation
loss. The parent is a checkpoint of any stage and model family (`trainer/checkpoints.py`;
usually SFT, which has learned the answer formats) on the run volume; its recorded
architecture picks the family adapter (`trainer/families.py`), which builds the model,
encodes prompts, samples, says where answers are and writes exports (at the end, and to
`hf_repo` if set). Replay is for the subword family.

A run saves a checkpoint every `save_every` steps, every `save_seconds` and when it stops
(the last `keep_checkpoints` are kept, all when 0): the weights, optimizer, step and update
counts, the prompt scheduler, the trajectory source and every random generator. Training
stops at the step target (`stop_reason: steps`, then a final validation and the family's
export), at `settings.max_seconds`, the absolute `deadline` or when `should_stop()`; a later
call with the same output folder resumes its latest checkpoint exactly, with the same
identity except the step target and the save, validation and time settings.
"""

import json
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

from training_data import rl_tasks, rl_trajectories
from training_data.subword import LOSS_IGNORE_INDEX, digest

from trainer import batches, checkpoints, optimizers
from trainer.batches import collate
from trainer.replay_hook import ReplayHook
from trainer.rl.batch import Batch, Tokens
from trainer.rl.off_policy.sources import Reuse, TrajectoryFile
from trainer.rl.off_policy.stable_reinforce import StableReinforce
from trainer.rl.ranks import Ranks
from trainer.rl.rollouts import Sampler, Scheduler, collect

FORMAT = "rl-v1"
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


def source_files(family):
    """{name: path} of the code a run depends on, hashed and copied into the run."""
    import training_data.sandbox.execution as execution_module
    import training_data.sandbox.runner as runner_module

    package = Path(__file__).parent
    return {
        **{
            "rl_" + "_".join(path.relative_to(package).with_suffix("").parts): str(path)
            for path in sorted(package.rglob("*.py"))
        },
        "batches": batches.__file__,
        "optimizers": optimizers.__file__,
        "checkpoints": checkpoints.__file__,
        "rl_tasks": rl_tasks.__file__,
        "rl_trajectories": rl_trajectories.__file__,
        "math_sft": sys.modules["training_data.math_sft"].__file__,
        "code_sft": sys.modules["training_data.code_sft"].__file__,
        "sandbox_execution": execution_module.__file__,
        "sandbox_runner": runner_module.__file__,
        **ReplayHook.SOURCES,
        **{name: str(path) for name, path in family.source_files().items()},
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
    trajectories_path=None,
    should_stop=lambda: False,
    ranks=None,
):
    """Train, or resume `output`'s latest checkpoint, until a stop (see the module
    docstring). `sandbox.run` checks code answers; `trajectories_path` is the trajectory
    file of the `"file"` source. With `ranks` (a process group per GPU), every rank draws
    the same prompts and samples and trains its share; gradients are averaged before
    clipping, and rank 0 logs, saves and exports."""
    import torch

    ranks = ranks or Ranks()
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing RL work")
    if not torch.cuda.is_available():
        raise RuntimeError("Training is CUDA/Modal only")
    algorithm, rollouts = settings.algorithm, settings.rollouts
    saved, description, family = checkpoints.parent(parent)
    if settings.replay and family.name != "subword":
        raise ValueError("Replay reads subword snapshots; it is for the subword family")
    if settings.hf_repo and not family.exports:
        raise ValueError(f"The {family.name} family writes no exports to publish")
    if settings.stable_reinforce and not family.records_sampler:
        raise ValueError(f"The {family.name} sampler records no log-probabilities")
    tokenizer = family.tokenizer()

    def with_prompt(row):
        return {**row, "prompt_ids": family.prompt_ids(tokenizer, row["prompt"])}

    pool, pool_hash = rl_tasks.load(pool_path)
    pool = {
        split: [r for r in map(with_prompt, rows) if len(r["prompt_ids"]) <= MAX_PROMPT_TOKENS]
        for split, rows in pool.items()
    }
    trajectory_hash = None
    if settings.trajectories.source == "file":
        written, trajectory_hash = rl_trajectories.load(trajectories_path)
        trajectories = []
        for row in map(with_prompt, written):
            row["completion_ids"] = tokenizer.encode(
                row["completion"], add_special_tokens=False
            ).ids
            # Answers the policy could have written within its prompt and token limits.
            if len(row["prompt_ids"]) <= MAX_PROMPT_TOKENS and len(
                row["completion_ids"]
            ) < rollouts.token_limit(row["task"]):
                trajectories.append(row)
    files = source_files(family)
    source_hashes = {name: digest(path) for name, path in files.items()}
    config = description.model_config
    model = family.build(config).cuda()
    family.load(model, saved["model"])
    reference = family.build(config).cuda().eval()
    family.load(reference, saved["model"])
    reference.requires_grad_(False)
    del saved
    torch.manual_seed(settings.seed + ranks.rank)
    # The same draws of prompts on every rank; each rank's own sampling.
    rng = random.Random(settings.seed)
    shuffler = random.Random(settings.seed + ranks.rank)
    generator = torch.Generator(device="cuda").manual_seed(settings.seed + ranks.rank)
    optimizer = settings.optimizer.make(model)
    replay = ReplayHook.attach(settings, data_root, family.context(model))
    stable = StableReinforce.attach(settings)
    identity = {
        "parent": str(parent),
        "parent_sha256": digest(parent),
        "parent_stage": description.stage,
        # The parent's model code, which may predate this trainer's (`source_sha256`).
        "parent_source_sha256": description.source_sha256,
        "family": family.name,
        "pool_sha256": pool_hash,
        "trajectories_sha256": trajectory_hash,
        "tokenizer_sha256": family.tokenizer_sha256,
        "replay": replay.identity() if replay else None,
        "settings": asdict(settings),
        "world_size": ranks.world,
        "source_sha256": source_hashes,
    }
    output = Path(output)
    resumed = latest_checkpoint(output)
    if resumed is None and ranks.main:
        # A run stopped before its first checkpoint starts again.
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(json.dumps(identity, indent=2))
        shutil.copyfile(pool_path, output / "pool.jsonl")
        family.copy_tokenizer(output / "tokenizer")
        for name, path in files.items():
            shutil.copyfile(path, output / f"{name}{Path(path).suffix}")
    ranks.barrier()

    def sample(rows):
        return collect(
            family,
            model,
            rows,
            rollouts,
            rollouts.temperature,
            generator,
            tokenizer,
            sandbox,
            stable.top_k if stable else 0,
        )

    scheduler = Scheduler(pool["train"], rollouts.rest_steps, rng)
    source = {
        "sampler": lambda: Sampler(rollouts, scheduler, sample, ranks),
        "reuse": lambda: Reuse(
            rollouts, scheduler, sample, settings.trajectories.updates_per_batch, ranks
        ),
        "file": lambda: TrajectoryFile(
            trajectories,
            settings.trajectories.answers_per_step,
            rng,
            family.stop_ids[0],
            ranks,
        ),
    }[settings.trajectories.source]()
    step, update, trained_tokens = 0, 0, 0
    if resumed is not None:
        # Only load our trusted private run-volume checkpoints.
        state = torch.load(resumed, map_location="cpu", weights_only=False)
        check_resume(state["identity"], identity)
        family.load(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, update, trained_tokens = state["step"], state["update"], state["trained_tokens"]
        scheduler.state = state["prompt_state"]
        if state["source_state"] is not None:
            source.load_state(state["source_state"])
        if replay:
            replay.load_state(state["replay_state"])
        rng.setstate(state["rng"])
        own = state["rank_rng"][ranks.rank]
        shuffler.setstate(own["shuffler"])
        torch.set_rng_state(own["torch"])
        torch.cuda.set_rng_state(own["cuda"])
        generator.set_state(own["generator"])
        del state

    def evaluate():
        metrics = {}
        mine = pool["validation"][ranks.rank :: ranks.world]
        local = collect(family, model, mine, rollouts, 0, generator, tokenizer, sandbox)
        batch = merged(local, ranks)
        for task in rl_tasks.TASKS:
            picked = [i for i, row in enumerate(batch.rows) if row["task"] == task]
            if picked:
                metrics[f"validation_{task}_accuracy"] = sum(
                    batch.rewards[i] for i in picked
                ) / len(picked)
                metrics[f"validation_{task}_truncated"] = 1 - sum(
                    batch.finished[i] for i in picked
                ) / len(picked)
        # Per dataset, named by the row ID's prefix (gsm8k, math, kodcode, mbpp, oci).
        datasets = {}
        for i, row in enumerate(batch.rows):
            datasets.setdefault(row["id"].split("-")[0], []).append(batch.rewards[i])
        if len(datasets) > 1:
            metrics["validation_accuracy_by_dataset"] = {
                name: sum(values) / len(values) for name, values in sorted(datasets.items())
            }
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
        optimizers.set_learning_rate(optimizer, settings.optimizer.factor(update))
        optimizer.zero_grad(set_to_none=True)
        # Dr. GRPO's constant: the answers per update times the task's token limit.
        per_update = len(batch.rows) / source.updates_per_batch
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
            tokens = Tokens(logits, y, mask, logp, old_logp.detach(), weight, scale)
            correction = stable.apply(tokens, batch, chosen) if stable else 0.0
            loss, stats = algorithm.loss(tokens)
            loss = loss + correction
            del logits, tokens
            delta = (reference_logp - logp) * mask
            kl = (delta.exp() - delta - 1) * mask
            if settings.kl_coef:
                loss = loss + settings.kl_coef * (scale[:, None] * kl).sum()
            loss.backward()
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value
            totals["kl"] += kl.detach().sum().item()
            totals["nll"] += -logp.detach().sum().item()
            totals["tokens"] += int(mask.sum())
        if replay:
            totals["replay_loss"] += replay.backward(model)
        ranks.average_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Non-finite RL gradient")
        # Gradients are averaged over ranks, so every rank steps if any rank trained.
        if ranks.sum(len(answers)) or replay:
            optimizer.step()
        totals["grad_norm"] += grad_norm.item()
        update += 1

    def save():
        """Every rank's generators, then rank 0 writes the checkpoint (all ranks call it)."""
        target = output / f"step-{step:09d}.pt"
        rank_rng = ranks.gather(
            [
                {
                    "shuffler": shuffler.getstate(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(),
                    "generator": generator.get_state(),
                }
            ]
        )
        if not ranks.main:
            return str(target)
        temporary = target.with_suffix(".partial")
        torch.save(
            {
                "format": FORMAT,
                "stage": "rl",
                "identity": identity,
                "special_tokens": family.special_tokens,
                "architecture": family.architecture(model),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "update": update,
                "trained_tokens": trained_tokens,
                "prompt_state": scheduler.state,
                "source_state": source.state() if hasattr(source, "state") else None,
                "replay_state": replay.state() if replay else None,
                "rng": rng.getstate(),
                "rank_rng": rank_rng,
            },
            temporary,
        )
        temporary.replace(target)
        if settings.keep_checkpoints:
            for old in sorted(output.glob("step-*.pt"))[: -settings.keep_checkpoints]:
                old.unlink()
        commit()
        return str(target)

    def record(**values):
        if ranks.main:
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **values}) + "\n")
            print(json.dumps({"step": step, **values}), flush=True)

    started = time.monotonic()
    if resumed is None:
        record(**evaluate(), phase="baseline")
        if ranks.main:
            commit()
    if ranks.main:
        print(json.dumps({"stage": "training", "step": step, "target": settings.steps}), flush=True)
    last_saved, saved_at = step if resumed is not None else -1, time.monotonic()

    def stop_reason():
        if step >= settings.steps:
            return "steps"
        if time.monotonic() - started >= settings.max_seconds:
            return "time_limit"
        if deadline is not None and time.time() >= deadline:
            return "deadline"
        if should_stop():
            return "stop_requested"
        return None

    shares = source.updates_per_batch
    while (reason := ranks.broadcast(stop_reason())) is None:
        timings = {}
        clock = time.monotonic()
        batch = source.batch(step)
        timings["generate_seconds"] = time.monotonic() - clock
        clock = time.monotonic()
        advantage = algorithm.advantages(batch)
        sequences = batch.sequences()
        trained = [
            i for i, a in enumerate(advantage) if a != 0 or not algorithm.skips_zero_advantage
        ]
        shuffler.shuffle(trained)
        old = old_logprobs(sequences, trained) if algorithm.needs_behaviour else None
        totals = {"kl": 0.0, "nll": 0.0, "tokens": 0, "grad_norm": 0.0, "replay_loss": 0.0}
        for share in range(shares):
            answers = trained[share * len(trained) // shares : (share + 1) * len(trained) // shares]
            optimize(batch, sequences, advantage, sorted(answers), old, totals)
        step += 1
        timings["train_seconds"] = time.monotonic() - clock
        totals = {key: ranks.sum(value) for key, value in totals.items()}
        trained_tokens += totals["tokens"]
        batch = merged(batch, ranks)
        values = {}
        for task in rl_tasks.TASKS:
            picked = [i for i, row in enumerate(batch.rows) if row["task"] == task]
            if not picked:
                continue
            values[f"{task}_reward"] = sum(batch.rewards[i] for i in picked) / len(picked)
            if batch.group:
                groups = [picked[k : k + batch.group] for k in range(0, len(picked), batch.group)]
                values[f"{task}_uniform_groups"] = sum(
                    len({batch.rewards[i] for i in g}) == 1 for g in groups
                ) / len(groups)
            values[f"{task}_answer_tokens"] = sum(len(batch.completions[i]) for i in picked) / len(
                picked
            )
            values[f"{task}_truncated"] = 1 - sum(batch.finished[i] for i in picked) / len(picked)
        tokens = max(1, totals.pop("tokens"))
        values.update(
            trained_answers=ranks.sum(len(trained)),
            sampled_nll=totals.pop("nll") / tokens,
            kl=totals.pop("kl") / tokens,
            **(stable.statistics() if stable else {}),
            # Every rank clips the same averaged gradients.
            grad_norm=totals.pop("grad_norm") / shares / ranks.world,
            lr=settings.optimizer.learning_rate(update - 1),
            updates=update,
            **source.statistics(step),
            **({"replay_loss": totals.pop("replay_loss") / shares / ranks.world} if replay else {}),
            **{key: value / tokens for key, value in totals.items() if key != "replay_loss"},
            **timings,
        )
        record(**values)
        if step % settings.evaluate_every == 0:
            record(**evaluate())
        if step % settings.save_every == 0 or time.monotonic() - saved_at >= settings.save_seconds:
            save()
            last_saved, saved_at = step, time.monotonic()
    if reason == "steps":
        record(**evaluate(), phase="final")
    path = str(output / f"step-{step:09d}.pt") if last_saved == step else save()
    if reason == "steps" and ranks.main:
        upload = family.export(
            model, output, {"step": step, "tokens": trained_tokens}, settings.hf_repo
        )
        if upload is not None:
            upload.join()
    if ranks.main:
        commit()
    ranks.barrier()
    return {
        "checkpoint": path,
        "step": step,
        "stop_reason": reason,
        "seconds": time.monotonic() - started,
    }


def merged(batch, ranks):
    """Every rank's answers of a `Batch` for statistics, in rank order (rows, completions,
    finished and rewards; groups stay whole)."""
    if ranks.world == 1:
        return batch
    parts = ranks.gather(
        [
            (
                [{k: r[k] for k in ("id", "task")} for r in batch.rows],
                batch.completions,
                batch.finished,
                batch.rewards,
            )
        ]
    )
    rows, completions, finished, rewards = (
        [x for part in parts for x in part[i]] for i in range(4)
    )
    return Batch(rows, completions, finished, rewards, [None] * len(rows), group=batch.group)


# Settings a resumed run may change: the step target and when it saves, validates, stops.
RESUMABLE = ("steps", "evaluate_every", "save_every", "save_seconds", "max_seconds")


def latest_checkpoint(output):
    """The run folder's newest `step-*.pt`, or None."""
    found = sorted(Path(output).glob("step-*.pt")) if Path(output).exists() else []
    return found[-1] if found else None


def check_resume(saved, current):
    """Refuse a resume whose identity differs beyond the `RESUMABLE` settings."""

    def fixed(identity):
        settings = {k: v for k, v in identity["settings"].items() if k not in RESUMABLE}
        return {**identity, "settings": settings}

    if fixed(saved) != fixed(current):
        changed = sorted(k for k in current if saved.get(k) != current[k])
        raise ValueError(f"Resume changed the run's identity: {changed}")
