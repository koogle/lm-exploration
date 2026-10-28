"""Distributed pretraining loop for the hybrid family (torchrun on one host).

Rank 0 reads the prepared-shard mixture and broadcasts each global batch; every
rank derives document ids and MTP targets from it, trains its slice with DDP
(bf16 autocast, fp32 parameters, torch.compile) and HybridOptimizer. The
objective is mean next-token loss plus the decaying MTP weight times mean MTP
loss, each normalized by its global target count. Learning rate follows
warmup-stable-decay over tokens. Rank 0 logs metrics.jsonl, evaluates held-out
loss and bits per byte per source, records gate and sink statistics, writes
fixed-prompt samples, and saves full checkpoints at a fixed interval and when the
stop time arrives. A run directory resumes from its latest checkpoint.

    torchrun --nproc_per_node=8 -m model_hybrid.train --config <run.json> \\
        --run-dir <dir> --stop-unix <time>
"""

import argparse
import contextlib
import dataclasses
import datetime
import faulthandler
import itertools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from trainer.optimizers import set_learning_rate

from . import documents as doc_utils
from . import model as hybrid
from . import optim
from . import tokenizer as glm

H100_BF16_FLOPS = 989e12
SAMPLE_PROMPTS = (
    "The history of the printing press",
    "def fibonacci(n):",
    "Question: A train travels 120 km in 2 hours. What is its average speed?\nAnswer:",
    "In mathematics, a prime number is",
)


@dataclasses.dataclass(frozen=True)
class RunConfig:
    recipe: str
    model: dict
    target_tokens: int
    global_rows: int = 256
    microbatch: int = 4
    warmup_tokens: int = 500_000_000
    decay_fraction: float = 0.15
    optimizer: dict = dataclasses.field(default_factory=dict)
    gradient_clip: float = 1.0
    compile: bool = True
    checkpoint_seconds: int = 5400
    keep_checkpoints: int = 2
    eval_every_steps: int = 500
    eval_documents: dict = dataclasses.field(
        default_factory=lambda: {"climbmix": 1000, "code": 300, "math": 300}
    )
    sample_every_steps: int = 2000
    log_every_steps: int = 10
    hf_repo: str = ""
    hf_every_tokens: int = 5_000_000_000
    seed: int = 1234
    # An export folder (config.json, model.safetensors) whose trunk weights start the
    # run when it has no checkpoint yet, e.g. a pretrained model for SFT.
    init_from: str = ""

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def learning_rate_factor(config, tokens):
    """Linear warmup, constant, then linear decay to 0 over the last decay_fraction."""
    if tokens < config.warmup_tokens:
        return tokens / config.warmup_tokens
    decay_start = (1 - config.decay_fraction) * config.target_tokens
    if tokens < decay_start:
        return 1.0
    return max(0.0, (config.target_tokens - tokens) / (config.target_tokens - decay_start))


def flops_per_token(model_config, lm, attended_keys):
    """Training FLOPs per token: 6 x matmul weights (incl. both output-layer uses) + attention.

    The MTP module reuses the output layer for its sampled fraction of positions.
    """
    matmul = sum(p.numel() for n, p in lm.named_parameters() if p.ndim == 2)
    if model_config.mtp:
        matmul += model_config.mtp_loss_fraction * lm.output_weight.numel()
    full_layers = model_config.layer_kinds.count("full") + (
        1 if model_config.mtp and model_config.mtp_layer_kind == "full" else 0
    )
    attention = 12 * full_layers * model_config.n_head * model_config.head_dim * attended_keys
    return 6 * matmul + attention


def mean_attended_keys(positions):
    return float((positions.float() + 1).mean())


class Runtime:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local)
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))
        self.device = torch.device("cuda", self.local)

    @property
    def main(self):
        return self.rank == 0


def model_config_of(config):
    model_config = hybrid.HybridConfig(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in config.model.items()}
    )
    if model_config.eos_id != glm.EOS_ID:
        raise ValueError("model eos_id must be the GLM tokenizer's EOS")
    return model_config


def build_model(config, device):
    model_config = model_config_of(config)
    lm = hybrid.HybridLM(model_config).to(device)
    return model_config, lm


def batch_tensors(batch, eos_id):
    inputs, labels = batch[0], batch[1]
    return inputs, labels, doc_utils.mtp_labels(inputs, labels, eos_id)


STOP_REASONS = ("", "stop_time", "data_exhausted", "data_late", "stop_requested")
HANG_DUMP_SECONDS = 1200
# The launcher creates this file when a graceful stop is requested.
STOP_FILE = os.environ.get("HYBRID_STOP_FILE", "/tmp/hybrid-stop")


def broadcast_batch(runtime, mixture, rows, context, stop_unix):
    """Rank 0 decides whether to continue and reads the batch; every rank gets both.

    Returns (int64 [2, rows, context] batch or None, row sources, stop reason).
    """
    control = torch.zeros(1, dtype=torch.int32, device=runtime.device)
    names, stacked = None, None
    if runtime.main:
        if time.time() >= stop_unix:
            control[0] = 1
        elif os.path.exists(STOP_FILE):
            control[0] = 4
        else:
            try:
                inputs, labels, names = mixture.next_batch(rows)
                stacked = np.stack((inputs, labels)).astype(np.int32)
            except StopIteration:
                control[0] = 2
            except TimeoutError:
                control[0] = 3
    dist.broadcast(control, 0)
    if control.item():
        return None, None, STOP_REASONS[control.item()]
    buffer = torch.empty(2, rows, context, dtype=torch.int32, device=runtime.device)
    if runtime.main:
        buffer.copy_(torch.from_numpy(stacked))
    dist.broadcast(buffer, 0)
    return buffer.long(), names, ""


def checkpoint_paths(run_dir):
    return sorted(Path(run_dir, "checkpoints").glob("step-*.pt"))


def save_checkpoint(run_dir, lm, optimizer, mixture_state, progress, keep):
    folder = Path(run_dir, "checkpoints")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"step-{progress['step']:07d}.pt"
    state = {
        "model": lm.state_dict(),
        "optimizer": optimizer.state_dict(),
        "data": mixture_state,
        "progress": progress,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        },
    }
    torch.save(state, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    (folder / "latest.json").write_text(json.dumps({"path": path.name, **progress}) + "\n")
    for old in checkpoint_paths(run_dir)[:-keep]:
        old.unlink()
    return path


def gate_sink_statistics(lm):
    """Parameter statistics of every attention sink, gate bias and KDA decay."""
    stats = {}
    for name, module in lm.named_modules():
        if isinstance(module, hybrid.Attention):
            sink = module.sink.detach().float()
            bias = module.gate_bias.detach().float()
            stats[name] = {
                "sink_mean": sink.mean().item(),
                "sink_max": sink.max().item(),
                "gate_bias_open": torch.sigmoid(bias).mean().item(),
            }
        elif name.endswith("mixer") and hasattr(module, "A_log"):
            stats[name] = {"decay_rate_mean": module.A_log.detach().float().exp().mean().item()}
    return stats


# In-document position buckets for long-context evaluation (used above 4K context).
POSITION_EDGES = (0, 1024, 4096, 16384, 32768)


def document_positions(inputs, eos_id):
    """Each input token's position within its document; EOS ends a document."""
    positions = np.zeros(inputs.shape, dtype=np.int64)
    for row, tokens in enumerate(inputs):
        position = 0
        for column, token in enumerate(tokens):
            positions[row, column] = position
            position = 0 if token == eos_id else position + 1
    return positions


def position_losses(lm, inputs, labels, device, eos_id, edges):
    """Held-out loss of the targets whose input sits in each in-document position bucket."""
    positions = document_positions(inputs, eos_id)
    out = {}
    for low, high in itertools.pairwise(edges):
        keep = (positions >= low) & (positions < high) & (labels != -100)
        if not keep.any():
            continue
        ce, count = 0.0, 0
        for start in range(0, len(inputs), 4):
            x = torch.from_numpy(inputs[start : start + 4]).to(device)
            y = torch.from_numpy(np.where(keep, labels, -100)[start : start + 4]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = lm(x, y)
            ce, count = ce + losses["ce"].item(), count + int(losses["count"])
        if count:
            out[f"{low}-{high}"] = ce / count
    return out


@torch.no_grad()
def evaluate(lm, heldout, device, eos_id):
    """Held-out loss, bits per byte and MTP loss per source; gate/sink activations; above
    4K context also the loss per in-document position bucket."""
    lm.eval()
    results = {}
    hybrid.Attention.record_statistics = True
    try:
        for name, (inputs, labels, total_bytes) in heldout.items():
            sums = {"ce": 0.0, "count": 0, "mtp_ce": 0.0, "mtp_count": 0}
            activations = []
            for start in range(0, len(inputs), 4):
                x = torch.from_numpy(inputs[start : start + 4]).to(device)
                y = torch.from_numpy(labels[start : start + 4]).to(device)
                targets = doc_utils.mtp_labels(x, y, eos_id) if lm.mtp is not None else None
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = lm(x, y, targets)
                losses.setdefault("mtp_ce", torch.zeros(()))
                losses.setdefault("mtp_count", torch.zeros((), dtype=torch.long))
                for key in sums:
                    sums[key] += losses[key].item() if key.endswith("ce") else int(losses[key])
                activations.append(
                    [m.statistics for m in lm.modules() if isinstance(m, hybrid.Attention)]
                )
            layers = list(zip(*activations, strict=True))
            results[name] = {
                "loss": sums["ce"] / sums["count"],
                "bits_per_byte": sums["ce"] / math.log(2) / total_bytes,
                "mtp_loss": sums["mtp_ce"] / max(sums["mtp_count"], 1),
                "tokens": sums["count"],
                "gate_open": [float(np.mean([s["gate_open"] for s in layer])) for layer in layers],
                "sink_weight": [
                    float(np.mean([s["sink_weight"] for s in layer])) for layer in layers
                ],
            }
            if lm.config.block_size > POSITION_EDGES[2]:
                edges = [e for e in POSITION_EDGES if e < lm.config.block_size]
                edges.append(lm.config.block_size)
                results[name]["loss_by_position"] = position_losses(
                    lm, inputs, labels, device, eos_id, edges
                )
    finally:
        hybrid.Attention.record_statistics = False
        lm.train()
    return results


@torch.no_grad()
def samples(lm, tokenizer, device, prompts=SAMPLE_PROMPTS, stop_ids=(), new_tokens=48):
    """Greedy continuations of fixed prompts (full recompute per token), up to a stop id."""
    lm.eval()
    outputs = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        start = len(ids)
        for _ in range(new_tokens):
            x = torch.tensor([ids], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids.append(int(lm(x)[0, -1].argmax()))
            if ids[-1] in stop_ids:
                break
        completion = tokenizer.decode(ids[start:], skip_special_tokens=False)
        outputs.append({"prompt": prompt, "completion": completion})
    lm.train()
    return outputs


def upload_export(lm, model_config, run_dir, progress, repo):
    """bf16 trunk weights, config and metrics to the HF model repo, in a background thread."""
    import threading

    from safetensors.torch import save_file

    folder = Path(run_dir, "export", f"tokens-{progress['tokens']:012d}")
    folder.mkdir(parents=True, exist_ok=True)
    weights = {
        k: v.detach().to("cpu", torch.bfloat16).contiguous()
        for k, v in lm.state_dict().items()
        if not k.startswith("mtp.")
    }
    save_file(weights, folder / "model.safetensors")
    config = dataclasses.asdict(model_config) | {
        "tokenizer": glm.SOURCE,
        "tokenizer_sha256": glm.SHA256,
    }
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (folder / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    metrics = Path(run_dir, "metrics.jsonl")

    def push():
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo=folder.name)
        if metrics.exists():
            api.upload_file(path_or_fileobj=metrics, path_in_repo="metrics.jsonl", repo_id=repo)

    thread = threading.Thread(target=push, daemon=False)
    thread.start()
    return thread


def train(
    config,
    run_dir,
    stop_unix,
    max_tokens=None,
    data_source=None,
    save_final=True,
    on_step=None,
    heldout_source=None,
    sample_prompts=SAMPLE_PROMPTS,
    sample_stop_ids=(),
    sample_new_tokens=48,
):
    """Train until target/max tokens, data exhaustion or stop_unix; returns final progress.

    `heldout_source(recipe, mixture, context)` (rank 0) returns the held-out
    windows {name: (inputs, labels, bytes)}; `on_step(progress)` runs on every rank
    after each optimizer step.
    """
    runtime = Runtime()
    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    run_dir = Path(run_dir)
    model_config, lm = build_model(config, runtime.device)
    settings = {k: tuple(v) if isinstance(v, list) else v for k, v in config.optimizer.items()}
    optimizer = optim.HybridOptimizer(lm, optim.OptimizerConfig(**settings), fused=True)
    context = model_config.block_size
    recipe = json.loads(Path(config.recipe).read_text()) if config.recipe else None
    mixture = data_source(recipe) if data_source else None
    progress = {"step": 0, "tokens": 0, "seconds": 0.0}
    latest = checkpoint_paths(run_dir)
    if latest:
        state = torch.load(latest[-1], map_location=runtime.device, weights_only=False)
        lm.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        progress = state["progress"]
        if mixture is not None and runtime.main:
            mixture.load_state_dict(state["data"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        torch.cuda.set_rng_state(state["rng"]["cuda"].cpu())
        del state
    elif config.init_from:
        from safetensors.torch import load_file

        lm.load_state_dict(load_file(Path(config.init_from) / "model.safetensors", device="cpu"))
    compiled = torch.compile(lm) if config.compile else lm
    ddp = DistributedDataParallel(compiled, device_ids=[runtime.local], broadcast_buffers=False)
    rows_per_rank = config.global_rows // runtime.world
    if rows_per_rank * runtime.world != config.global_rows or rows_per_rank % config.microbatch:
        raise ValueError("global_rows must split evenly into ranks and microbatches")
    accumulation = rows_per_rank // config.microbatch
    tokens_per_step = config.global_rows * context
    limit = min(config.target_tokens, max_tokens or config.target_tokens)
    tokenizer = heldout = None
    if runtime.main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(dataclasses.asdict(config), indent=2) + "\n"
        )
        if recipe and heldout_source:
            heldout = heldout_source(recipe, mixture, context)
            tokenizer = glm.load()
    metrics_file = open(run_dir / "metrics.jsonl", "a") if runtime.main else None  # noqa: SIM115
    last_checkpoint, uploads = time.monotonic(), []
    next_upload = (progress["tokens"] // config.hf_every_tokens + 1) * config.hf_every_tokens
    params = sum(p.numel() for p in lm.parameters())
    stop_reason = "target" if limit == config.target_tokens else "token_limit"

    def checkpoint():
        if runtime.main:
            state = mixture.state_dict() if mixture is not None else None
            save_checkpoint(run_dir, lm, optimizer, state, dict(progress), config.keep_checkpoints)

    while progress["tokens"] + tokens_per_step <= limit:
        # A step that takes this long is stuck: print every thread's stack (repeatedly).
        faulthandler.dump_traceback_later(HANG_DUMP_SECONDS, repeat=True)
        start = time.perf_counter()
        batch, names, reason = broadcast_batch(
            runtime, mixture, config.global_rows, context, stop_unix
        )
        if reason:
            stop_reason = reason
            break
        inputs, labels, mtp_labels = batch_tensors(batch, model_config.eos_id)
        ntp_count = (labels != -100).sum()
        # Expected number of sampled valid MTP targets (see HybridLM.forward).
        mtp_count = (mtp_labels != -100).sum().clamp_min(1) * model_config.mtp_loss_fraction
        fraction = progress["tokens"] / config.target_tokens
        mtp_weight = hybrid.mtp_weight(model_config, fraction) if model_config.mtp else 0.0
        lr_factor = learning_rate_factor(config, progress["tokens"] + tokens_per_step)
        set_learning_rate(optimizer, lr_factor * optimizer.base_lr)
        progress["lr_factor"] = lr_factor
        sums = torch.zeros(3, device=runtime.device)
        mine = slice(runtime.rank * rows_per_rank, (runtime.rank + 1) * rows_per_rank)
        for micro in range(accumulation):
            rows = slice(
                mine.start + micro * config.microbatch, mine.start + (micro + 1) * config.microbatch
            )
            sync = micro == accumulation - 1
            with contextlib.nullcontext() if sync else ddp.no_sync():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = ddp(
                        inputs[rows],
                        labels[rows],
                        mtp_labels[rows] if model_config.mtp else None,
                    )
                objective = losses["loss"] / ntp_count
                if model_config.mtp:
                    objective = objective + mtp_weight * losses["mtp_loss"] / mtp_count
                (objective * runtime.world).backward()
            sums += torch.stack(
                (
                    losses["ce"].detach(),
                    losses["z"].detach(),
                    losses["mtp_ce"].detach()
                    if model_config.mtp
                    else torch.zeros((), device=runtime.device),
                )
            )
        dist.all_reduce(sums)
        norm = torch.nn.utils.clip_grad_norm_(lm.parameters(), config.gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss = (sums[0] / ntp_count).item()
        if not math.isfinite(loss) or not math.isfinite(norm.item()):
            stop_reason = "non_finite"
            break
        elapsed = time.perf_counter() - start
        progress["step"] += 1
        progress["tokens"] += tokens_per_step
        progress["seconds"] += elapsed
        if on_step:
            on_step(progress)
        if runtime.main:
            record = None
            if progress["step"] % config.log_every_steps == 0 or progress["step"] == 1:
                documents = doc_utils.document_ids(inputs[:8], model_config.eos_id)
                keys = mean_attended_keys(doc_utils.document_positions(documents))
                rate = tokens_per_step / elapsed
                record = {
                    "step": progress["step"],
                    "tokens": progress["tokens"],
                    "loss": loss,
                    "z": (sums[1] / ntp_count).item(),
                    "mtp_loss": (sums[2] / mtp_count).item() if model_config.mtp else None,
                    "mtp_weight": mtp_weight,
                    "lr_factor": lr_factor,
                    "grad_norm": norm.item(),
                    "tokens_per_second": rate,
                    "mfu": rate
                    * flops_per_token(model_config, lm, keys)
                    / (runtime.world * H100_BF16_FLOPS),
                    "sources": {n: names.count(n) for n in sorted(set(names))} if names else None,
                }
                print(json.dumps(record), flush=True)
            if heldout and progress["step"] % config.eval_every_steps == 0:
                record = (record or {"step": progress["step"], "tokens": progress["tokens"]}) | {
                    "heldout": evaluate(lm, heldout, runtime.device, model_config.eos_id),
                    "parameters": gate_sink_statistics(lm),
                }
            if tokenizer and progress["step"] % config.sample_every_steps == 0:
                record = (record or {"step": progress["step"]}) | {
                    "samples": samples(
                        lm,
                        tokenizer,
                        runtime.device,
                        sample_prompts,
                        sample_stop_ids,
                        sample_new_tokens,
                    )
                }
            if record:
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
        if time.monotonic() - last_checkpoint >= config.checkpoint_seconds:
            checkpoint()
            last_checkpoint = time.monotonic()
        if runtime.main and config.hf_repo and progress["tokens"] >= next_upload:
            uploads.append(upload_export(lm, model_config, run_dir, dict(progress), config.hf_repo))
            next_upload += config.hf_every_tokens
    if save_final and stop_reason != "non_finite":
        checkpoint()
    if runtime.main:
        if heldout:
            final = {
                "step": progress["step"],
                "tokens": progress["tokens"],
                "final": True,
                "heldout": evaluate(lm, heldout, runtime.device, model_config.eos_id),
                "parameters": gate_sink_statistics(lm),
            }
            if tokenizer:
                final["samples"] = samples(
                    lm,
                    tokenizer,
                    runtime.device,
                    sample_prompts,
                    sample_stop_ids,
                    sample_new_tokens,
                )
            metrics_file.write(json.dumps(final) + "\n")
            metrics_file.close()
        if config.hf_repo and stop_reason == "target":
            uploads.append(upload_export(lm, model_config, run_dir, dict(progress), config.hf_repo))
        for thread in uploads:
            thread.join()
        summary = {**progress, "stop_reason": stop_reason, "parameters": params}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"summary": summary}), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return progress


def heldout_windows(recipe, root, name, context, max_documents):
    """Held-out windows of a pretraining source or a chat subset: (inputs, labels, bytes).

    Pretraining sources concatenate whole documents into one stream
    (`packed_shards.heldout_windows`); chat subsets pack whole conversations and train
    only assistant tokens (`sft_data.chat_heldout_windows`, None when empty).
    """
    if recipe.get("format") == "chat-sft":
        from .sft_data import chat_heldout_windows

        return chat_heldout_windows(recipe, root, name, context, max_documents)
    from training_data.packed_shards import heldout_windows as source_windows

    return source_windows(recipe, root, name, context, max_documents)


def pretraining_heldout(config):
    def heldout(recipe, mixture, context):
        from training_data.packed_shards import heldout_index, shard_paths

        for name, source in recipe["sources"].items():
            meta = shard_paths(mixture.root, name, heldout_index(source))[2]
            waited = time.monotonic()
            while not meta.exists():
                if time.monotonic() - waited > mixture.wait_seconds:
                    raise TimeoutError(f"Held-out shard for {name} was not prepared")
                time.sleep(15)
        return {
            name: heldout_windows(recipe, mixture.root, name, context, count)
            for name, count in config.eval_documents.items()
            if name in recipe["sources"]
        }

    return heldout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stop-unix", type=float, required=True)
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()
    config = RunConfig.load(args.config)
    recipe = json.loads(Path(config.recipe).read_text())
    root = Path(args.data_root) / recipe["name"]
    from training_data.packed_shards import PreparedMixture, ReplayMixture

    # Replay of an earlier recipe's prepared shards, read at this recipe's context.
    replay = recipe.get("replay")
    if replay:
        replay_recipe = json.loads(
            (Path(config.recipe).parent / f"{replay['recipe']}.json").read_text()
        )
        replay_recipe["context"] = recipe["context"]
        replay_root = Path(args.data_root) / replay_recipe["name"]
        replay_rows = round(config.global_rows * replay["fraction"])

    def with_replay(primary):
        if not replay:
            return primary
        return ReplayMixture(primary, PreparedMixture(replay_recipe, replay_root), replay_rows)

    def replay_heldout(context):
        return {
            name: heldout_windows(replay_recipe, replay_root, name, context, count)
            for name, count in config.eval_documents.items()
            if replay and name in replay_recipe["sources"]
        }

    if recipe.get("format") == "chat-sft":
        from . import sft_data

        def heldout_source(recipe, mixture, context):
            """Chat held-out per file, plus the replayed sources' held-out shards."""
            count = config.eval_documents["chat"]
            out = {
                name: windows
                for name in sft_data.subset_names(recipe)
                if (windows := heldout_windows(recipe, root, name, context, count)) is not None
            }
            return out | replay_heldout(context)

        options = {
            "data_source": lambda recipe: with_replay(
                sft_data.ChatMixture(recipe, root, config.seed)
            ),
            "heldout_source": heldout_source,
            "sample_prompts": sft_data.sample_prompts(),
            "sample_stop_ids": sft_data.STOP_IDS,
            "sample_new_tokens": 160,
        }
    else:
        own = pretraining_heldout(config)
        options = {
            "data_source": lambda recipe: with_replay(PreparedMixture(recipe, root)),
            "heldout_source": lambda recipe, mixture, context: (
                own(recipe, getattr(mixture, "primary", mixture), context) | replay_heldout(context)
            ),
        }
    train(config, args.run_dir, args.stop_unix, args.max_tokens, **options)


if __name__ == "__main__":
    main()
