"""RL with verifiable rewards (GRPO family) for a hybrid chat export, under torchrun.

Each step every rank takes `prompts_per_rank` prompts from the recipe's training
set (rl_data.PromptStream, the same seeded order on every rank), samples
`group_size` completions per prompt with a bf16 copy of the current policy
(generate.generate_batch, chat format, stopping at the end of the assistant turn)
and scores them (rewards.score: \\boxed{} answers by math_verify, code by its
pytest file). Advantages are group-relative without std scaling: reward minus
the group's mean reward (Dr. GRPO). Training is fully on-policy, one optimizer
step per batch, so the objective is the policy gradient itself,

    loss = -sum_i A_i * sum_t log p(y_it) / (completions * max_new_tokens),

summed over every completion token including the stop token, with a constant
normalizer so long and short completions are weighted by their tokens. No KL
term. Every `eval_every_steps` steps rank-sharded greedy completions of the
held-out prompts give the solve rate per source. Checkpoints (policy, optimizer,
prompt position, progress) are written every `checkpoint_seconds` and at stop;
the run resumes from the latest. With `hf_repo`, the bf16 trunk goes to that
repo at the end.
"""

import argparse
import contextlib
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from trainer.optimizers import set_learning_rate

from . import chat, generate, optim, rewards, rl_data, sft_data
from . import tokenizer as glm
from .model import HybridLM
from .train import STOP_FILE, Runtime, checkpoint_paths, model_config_of, upload_export


@dataclasses.dataclass(frozen=True)
class RLConfig:
    recipe: str
    model: dict
    init_from: str
    steps: int
    prompts_per_rank: int = 16
    group_size: int = 8
    max_new_tokens: int = 768
    temperature: float = 1.0
    microbatch: int = 8
    optimizer: dict = dataclasses.field(default_factory=dict)
    warmup_steps: int = 10
    gradient_clip: float = 1.0
    eval_every_steps: int = 50
    eval_prompts: int = 400
    checkpoint_seconds: int = 3600
    keep_checkpoints: int = 1
    log_every_steps: int = 1
    hf_repo: str = ""
    seed: int = 1234

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class Policy(nn.Module):
    """Weighted completion log-likelihood, so DDP can wrap the custom objective."""

    def __init__(self, lm, chunk_size=2048):
        super().__init__()
        self.lm, self.chunk_size = lm, chunk_size

    def forward(self, tokens, targets, weights):
        """Sums over target positions: weights * log p(target), and log p(target)."""
        hidden = self.lm.hidden_states(tokens)
        positions = targets.ne(-100)
        hidden, targets, weights = hidden[positions], targets[positions], weights[positions]
        weighted = logprob = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, targets.numel(), self.chunk_size):
            piece = slice(start, start + self.chunk_size)
            values = checkpoint(
                _target_logprobs,
                hidden[piece],
                self.lm.output_weight,
                targets[piece],
                use_reentrant=False,
            )
            weighted = weighted + (values * weights[piece]).sum()
            logprob = logprob + values.detach().sum()
        return weighted, logprob


def _target_logprobs(hidden, weight, targets):
    logits = F.linear(hidden, weight).float()
    return -F.cross_entropy(logits, targets, reduction="none")


def render_prompts(tokenizer, items):
    return [
        tokenizer.encode(
            chat.render(item["messages"], None, add_generation_prompt=True),
            add_special_tokens=False,
        ).ids
        for item in items
    ]


def completion_text(tokenizer, ids):
    return tokenizer.decode([t for t in ids if t not in sft_data.STOP_IDS])


def batch(prompts, completions, advantages, eos_id, multiple=256):
    """Padded (tokens, targets, weights) rows: targets on completion tokens only."""
    length = max(len(p) + len(c) for p, c in zip(prompts, completions, strict=True)) - 1
    length = -(-length // multiple) * multiple
    rows = len(prompts)
    tokens = torch.full((rows, length), eos_id, dtype=torch.long)
    targets = torch.full((rows, length), -100, dtype=torch.long)
    weights = torch.zeros((rows, length), dtype=torch.float32)
    for row, (prompt, completion, advantage) in enumerate(
        zip(prompts, completions, advantages, strict=True)
    ):
        sequence = prompt + completion
        tokens[row, : len(sequence) - 1] = torch.tensor(sequence[:-1])
        span = slice(len(prompt) - 1, len(sequence) - 1)
        targets[row, span] = torch.tensor(completion)
        weights[row, span] = advantage
    return tokens, targets, weights


def save(run_dir, lm, optimizer, stream, progress, keep):
    folder = Path(run_dir, "checkpoints")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"step-{progress['step']:07d}.pt"
    state = {
        "model": lm.state_dict(),
        "optimizer": optimizer.state_dict(),
        "prompts": stream.state_dict(),
        "progress": progress,
    }
    torch.save(state, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    for old in checkpoint_paths(run_dir)[:-keep]:
        old.unlink()


@torch.no_grad()
def evaluate(runtime, generator_model, tokenizer, heldout, max_new_tokens):
    """Greedy solve rate per source on the held-out prompts (sharded over ranks)."""
    mine = heldout[runtime.rank :: runtime.world]
    outputs = (
        generate.generate_batch(
            generator_model, render_prompts(tokenizer, mine), max_new_tokens, sft_data.STOP_IDS
        )
        if mine
        else []
    )
    scores = rewards.score(
        [
            (item["kind"], completion_text(tokenizer, ids), item["reference"])
            for item, ids in zip(mine, outputs, strict=True)
        ]
    )
    sources = sorted({item["source"] for item in heldout})
    totals = torch.zeros(2, len(sources), device=runtime.device)
    for item, value in zip(mine, scores, strict=True):
        index = sources.index(item["source"])
        totals[0, index] += value
        totals[1, index] += 1
    dist.all_reduce(totals)
    return {s: (totals[0, i] / totals[1, i]).item() for i, s in enumerate(sources)}


def train(config, run_dir, root, stop_unix):
    runtime = Runtime()
    torch.manual_seed(config.seed + runtime.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    run_dir = Path(run_dir)
    recipe = json.loads(Path(config.recipe).read_text())
    folder = Path(root) / recipe["name"]
    stream = rl_data.PromptStream(rl_data.load(folder / "prompts.jsonl"), config.seed)
    heldout = rl_data.load(folder / "heldout.jsonl")[: config.eval_prompts]
    tokenizer = glm.load()
    model_config = model_config_of(config)
    lm = HybridLM(model_config).to(runtime.device)
    settings = {k: tuple(v) if isinstance(v, list) else v for k, v in config.optimizer.items()}
    optimizer = optim.HybridOptimizer(lm, optim.OptimizerConfig(**settings), fused=True)
    progress = {"step": 0, "tokens": 0, "seconds": 0.0}
    latest = checkpoint_paths(run_dir)
    if latest:
        state = torch.load(latest[-1], map_location=runtime.device, weights_only=False)
        lm.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        stream.load_state_dict(state["prompts"])
        progress = state["progress"]
        del state
    else:
        from safetensors.torch import load_file

        lm.load_state_dict(load_file(Path(config.init_from) / "model.safetensors", device="cpu"))
    policy = DistributedDataParallel(Policy(lm), device_ids=[runtime.local])
    sampler = HybridLM(model_config).to(runtime.device, torch.bfloat16).eval()
    generator = torch.Generator(device=runtime.device).manual_seed(
        config.seed * 1000 + runtime.rank + progress["step"]
    )
    completions_global = config.prompts_per_rank * config.group_size * runtime.world
    normalizer = completions_global * config.max_new_tokens
    if runtime.main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(dataclasses.asdict(config), indent=2) + "\n"
        )
    metrics = open(run_dir / "metrics.jsonl", "a") if runtime.main else None  # noqa: SIM115
    last_checkpoint, stop_reason = time.monotonic(), "steps"
    while progress["step"] < config.steps:
        stop = torch.tensor(
            [time.time() >= stop_unix or os.path.exists(STOP_FILE)], device=runtime.device
        )
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            stop_reason = "stop_time_or_request"
            break
        started = time.perf_counter()
        sampler.load_state_dict(lm.state_dict())
        items = stream.take(config.prompts_per_rank * runtime.world)
        items = items[runtime.rank :: runtime.world]
        prompts = render_prompts(tokenizer, items)
        repeated = [p for p in prompts for _ in range(config.group_size)]
        outputs = generate.generate_batch(
            sampler,
            repeated,
            config.max_new_tokens,
            sft_data.STOP_IDS,
            temperature=config.temperature,
            generator=generator,
        )
        generated = time.perf_counter()
        owners = [item for item in items for _ in range(config.group_size)]
        scores = rewards.score(
            [
                (item["kind"], completion_text(tokenizer, ids), item["reference"])
                for item, ids in zip(owners, outputs, strict=True)
            ]
        )
        scored = time.perf_counter()
        groups = torch.tensor(scores).view(-1, config.group_size)
        advantages = (groups - groups.mean(1, keepdim=True)).flatten().tolist()
        factor = min(1.0, (progress["step"] + 1) / max(1, config.warmup_steps))
        set_learning_rate(optimizer, factor * optimizer.base_lr)
        logprob_sum = torch.zeros((), device=runtime.device)
        order = range(0, len(repeated), config.microbatch)
        for index, start in enumerate(order):
            rows = slice(start, start + config.microbatch)
            tokens, targets, weights = batch(
                repeated[rows], outputs[rows], advantages[rows], model_config.eos_id
            )
            sync = index == len(order) - 1
            with contextlib.nullcontext() if sync else policy.no_sync():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    weighted, logprob = policy(
                        tokens.to(runtime.device),
                        targets.to(runtime.device),
                        weights.to(runtime.device),
                    )
                (-weighted / normalizer * runtime.world).backward()
            logprob_sum += logprob
        norm = torch.nn.utils.clip_grad_norm_(lm.parameters(), config.gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        progress["step"] += 1
        progress["seconds"] += time.perf_counter() - started
        lengths = torch.tensor([len(o) for o in outputs], dtype=torch.float32)
        stopped = torch.tensor([bool(o) and o[-1] in sft_data.STOP_IDS for o in outputs])
        sources = sorted(recipe["sources"])
        stats = torch.zeros(4 + 2 * len(sources), device=runtime.device)
        stats[0], stats[1] = lengths.sum(), stopped.float().sum()
        stats[2] = float((groups.std(1) > 0).sum())
        stats[3] = len(outputs)
        for item, value in zip(owners, scores, strict=True):
            stats[4 + 2 * sources.index(item["source"])] += value
            stats[5 + 2 * sources.index(item["source"])] += 1
        dist.all_reduce(stats)
        dist.all_reduce(logprob_sum)
        progress["tokens"] += int(stats[0].item())  # Completion tokens trained on.
        record = None
        if runtime.main and progress["step"] % config.log_every_steps == 0:
            count = stats[3].item()
            record = {
                "step": progress["step"],
                "reward": {
                    s: (stats[4 + 2 * i] / stats[5 + 2 * i]).item()
                    for i, s in enumerate(sources)
                    if stats[5 + 2 * i] > 0
                },
                "mean_length": (stats[0] / count).item(),
                "stopped": (stats[1] / count).item(),
                "groups_with_signal": (stats[2] / (count / config.group_size)).item(),
                "logprob_per_token": (logprob_sum / stats[0]).item(),
                "grad_norm": norm.item(),
                "lr_factor": factor,
                "seconds": {
                    "generate": generated - started,
                    "score": scored - generated,
                    "step": time.perf_counter() - started,
                },
            }
        if progress["step"] % config.eval_every_steps == 0 or progress["step"] == config.steps:
            sampler.load_state_dict(lm.state_dict())
            solved = evaluate(runtime, sampler, tokenizer, heldout, config.max_new_tokens)
            if runtime.main:
                record = (record or {"step": progress["step"]}) | {"heldout_solved": solved}
        if record:
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            print(json.dumps(record), flush=True)
        if time.monotonic() - last_checkpoint > config.checkpoint_seconds:
            if runtime.main:
                save(run_dir, lm, optimizer, stream, progress, config.keep_checkpoints)
            last_checkpoint = time.monotonic()
        if not math.isfinite(norm.item()):
            stop_reason = "non_finite"
            break
    if runtime.main:
        save(run_dir, lm, optimizer, stream, progress, config.keep_checkpoints)
        if config.hf_repo and stop_reason == "steps":
            upload_export(lm, model_config, run_dir, dict(progress), config.hf_repo).join()
        summary = {**progress, "stop_reason": stop_reason}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        metrics.close()
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stop-unix", type=float, required=True)
    args = parser.parse_args()
    train(RLConfig.load(args.config), args.run_dir, args.data_root, args.stop_unix)


if __name__ == "__main__":
    main()
