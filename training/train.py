"""Train the one architecture in 100m/model.py: uv run python train.py.

Edit TrainConfig below to change training settings; model dimensions belong in
100m/model.py. --config is an optional JSON settings file used by the local UI.
--resume optionally continues a trusted checkpoint; steps is the TOTAL target.
FP32 ~100M training needs several GiB; CPU can be slow. No run starts on import.
"""

import argparse
import json
import math
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from model_100m import model as architecture
from model_100m.model import GPT, ModelConfig
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from training_data.loading import get_batch, load_data

from run_utils import (
    atomic_write,
    load_checkpoint,
    restore_rng,
    rng_state,
    runtime_provenance,
    stop_at_step_boundary,
    write_json,
)

ROOT = Path(__file__).resolve().parent


@dataclass
class TrainConfig:
    # All normal training choices are here, not a long command line.
    data: str = str(ROOT / "data" / "tinyshakespeare.txt")
    out_dir: str | None = None  # A new runs/manual-* directory for each invocation.
    steps: int = 100
    batch_size: int = 1
    lr: float = 3e-4
    seed: int = 1337
    device: str = "cpu"
    eval_interval: int = 10
    log_interval: int = 1
    eval_batches: int = 2
    checkpoint_interval: int = 10
    resume: str | None = None


@torch.no_grad()
def evaluate(model, data, args):
    # Fixed held-out windows, independent of the training batch RNG.
    generator = torch.Generator().manual_seed(args.seed + 1)
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        tokens, targets = get_batch(
            data, args.batch_size, model.config.block_size, generator, args.device
        )
        logits = model(tokens)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_tokens(model, tokenizer, prompt, max_new_tokens, seed):
    # Yields each new token id as soon as it is sampled, so callers can stream output.
    context = tokenizer.encode(prompt) or [0]
    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            tokens = torch.tensor([context[-model.config.block_size :]], device=device)
            logits = model(tokens)
            probabilities = logits[0, -1].softmax(dim=-1).cpu()
            token = torch.multinomial(probabilities, 1, generator=generator).item()
            context.append(token)
            yield token
    finally:
        model.train(was_training)


def sample(model, tokenizer, prompt, max_new_tokens, seed):
    generated = list(sample_tokens(model, tokenizer, prompt, max_new_tokens, seed))
    return prompt + tokenizer.decode(generated)


def train(args, *, model_config=None):
    # model_config is an internal tiny-test seam, never a CLI or UI setting.
    validate_train_args(args)
    if args.out_dir is None:
        (ROOT / "runs").mkdir(exist_ok=True)
        args = replace(args, out_dir=tempfile.mkdtemp(prefix="manual-", dir=ROOT / "runs"))
    out = Path(args.out_dir)
    if not args.resume and any(
        (out / name).exists() for name in ("config.json", "metrics.jsonl", "checkpoint.pt")
    ):
        raise ValueError(
            "output directory already has a run; use --resume or choose a new directory"
        )
    with stop_at_step_boundary() as stop:
        block_size = model_config.block_size if model_config else ModelConfig.block_size
        tokenizer, train_data, val_data, fingerprint = load_data(args.data, block_size)
        model_config = model_config or ModelConfig(vocab_size=len(tokenizer.characters))
        if model_config.vocab_size != len(tokenizer.characters):
            raise ValueError("model vocabulary must match the dataset tokenizer")
        settings = {
            name: getattr(args, name)
            for name in (
                "steps",
                "batch_size",
                "lr",
                "seed",
                "device",
                "eval_interval",
                "log_interval",
                "eval_batches",
                "checkpoint_interval",
            )
        }
        settings.update(
            dtype="float32",
            optimizer="AdamW",
            betas=[0.9, 0.95],
            weight_decay=0.1,
            gradient_clip_norm=1.0,
        )
        config = {
            "model_name": "100m",
            "model": asdict(model_config),
            "training": settings,
            "tokenizer": tokenizer.config(),
            "dataset": {"path": str(Path(args.data).resolve()), **fingerprint},
            "runtime": runtime_provenance(args.device, architecture.__file__),
        }
        checkpoint = load_checkpoint(args.resume) if args.resume else None
        if checkpoint:
            old = checkpoint["config"]
            compatible = (
                checkpoint["dataset_fingerprint"] == fingerprint
                and old["model"] == config["model"]
                and old["tokenizer"] == config["tokenizer"]
            )
            fixed = (
                "batch_size",
                "lr",
                "seed",
                "device",
                "dtype",
                "optimizer",
                "betas",
                "weight_decay",
                "gradient_clip_norm",
            )
            if not compatible or any(old["training"][key] != settings[key] for key in fixed):
                raise ValueError(
                    "resume requires identical data, tokenizer, model and optimizer/batch/seed/device settings"
                )
            if args.steps < checkpoint["step"]:
                raise ValueError("steps cannot be smaller than the checkpoint's completed step")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        batch_generator = torch.Generator().manual_seed(args.seed)
        model = GPT(model_config).to(device=args.device, dtype=torch.float32)
        config["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        step, elapsed_before, last_loss = 0, 0.0, None
        if checkpoint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            step, elapsed_before, last_loss = (
                checkpoint[key] for key in ("step", "elapsed_seconds", "train_loss")
            )
            restore_rng(checkpoint["rng"], batch_generator)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "config.json", config)
        print(
            f"Model: {config['parameter_count']:,} parameters; device: {args.device}; FP32",
            flush=True,
        )
        started = time.perf_counter()

        def save():
            state = {
                "format_version": 1,
                "config": config,
                "model": model.state_dict(),
                "tokenizer": tokenizer.config(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "rng": rng_state(batch_generator),
                "dataset_fingerprint": fingerprint,
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                "train_loss": last_loss,
            }
            atomic_write(out / "checkpoint.pt", lambda stream: torch.save(state, stream))

        # Progress bar only in an interactive shell; the UI captures plain JSON lines.
        progress = tqdm(
            total=args.steps,
            initial=step,
            unit="step",
            dynamic_ncols=True,
            disable=not sys.stdout.isatty(),
        )
        with progress, (out / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
            while step < args.steps and not stop[0]:
                model.train()
                x, y = get_batch(
                    train_data,
                    args.batch_size,
                    model_config.block_size,
                    batch_generator,
                    args.device,
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss; no optimizer update performed")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                last_loss = loss.item()
                progress.update()
                final = step == args.steps or stop[0]
                if step % args.log_interval == 0 or step % args.eval_interval == 0 or final:
                    record = {
                        "step": step,
                        "train_loss": last_loss,
                        "tokens_seen": step * args.batch_size * model_config.block_size,
                    }
                    if step % args.eval_interval == 0 or final:
                        record["val_loss"] = evaluate(model, val_data, args)
                    record["elapsed_seconds"] = elapsed_before + time.perf_counter() - started
                    record["tokens_per_second"] = record["tokens_seen"] / record["elapsed_seconds"]
                    line = json.dumps(record, allow_nan=False)
                    metrics.write(line + "\n")
                    metrics.flush()
                    if progress.disable:
                        print(line, flush=True)
                    else:
                        progress.set_postfix(
                            loss=f"{last_loss:.3f}", tok_s=f"{record['tokens_per_second']:,.0f}"
                        )
                        if "val_loss" in record:
                            progress.write(
                                f"step {step:>6}  train {last_loss:.4f}  val {record['val_loss']:.4f}"
                            )
                if step % args.checkpoint_interval == 0:
                    save()
        save()
        text = sample(model, tokenizer, "", 80, args.seed + 2)
        atomic_write(out / "sample.txt", lambda stream: stream.write(text.encode("utf-8")))
        print(
            f"{'Stopped' if stop[0] else 'Completed'} at step {step}; "
            f"{config['parameter_count']} parameters; checkpoint: {out / 'checkpoint.pt'}",
            flush=True,
        )


def validate_train_args(args):
    for name in (
        "steps",
        "batch_size",
        "eval_interval",
        "log_interval",
        "eval_batches",
        "checkpoint_interval",
    ):
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63 - 2:
        raise ValueError("seed must be between 0 and 2**63 - 3")
    if args.device not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be cpu, mps or cuda")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="optional JSON training settings (used by UI)")
    parser.add_argument("--resume", help="trusted checkpoint to continue")
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    try:
        config = TrainConfig()
        if options.config:
            config = TrainConfig(**json.loads(options.config.read_text(encoding="utf-8")))
        if options.resume:
            config = replace(config, resume=options.resume)
        train(config)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
