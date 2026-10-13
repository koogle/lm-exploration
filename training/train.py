"""Training and generation; the architecture lives in 100m/model.py.

From training/: uv sync --locked; uv run lm-train self-test; uv run lm-ui.
Training: uv run lm-train train --data data/toy.txt --out-dir runs/demo.
Defaults target ~100M parameters in FP32 and need several GiB; CPU can be slow.
For cheap smoke tests override --n-layer 1 --n-head 2 --n-embd 16 --block-size 16.
--steps is the TOTAL update target; --resume requires identical data/settings.
Only load trusted checkpoints. No training starts on import.
"""

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from model_100m import model as architecture
from model_100m.model import GPT, ModelConfig
from torch import nn
from tqdm import tqdm
from training_data.loading import CharacterTokenizer, get_batch, load_data

from run_utils import (
    atomic_write,
    load_checkpoint,
    restore_rng,
    rng_state,
    runtime_provenance,
    stop_at_step_boundary,
    write_json,
)


@torch.no_grad()
def evaluate(model, data, args):
    # Fixed held-out windows, independent of the training batch RNG.
    generator = torch.Generator().manual_seed(args.seed + 1)
    model.eval()
    losses = [
        model(*get_batch(data, args.batch_size, args.block_size, generator, args.device))[1].item()
        for _ in range(args.eval_batches)
    ]
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model, tokenizer, prompt, max_new_tokens, seed):
    context = tokenizer.encode(prompt) or [0]
    generated = []
    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        tokens = torch.tensor([context[-model.config.block_size :]], device=device)
        logits, _ = model(tokens)
        probabilities = logits[0, -1].softmax(dim=-1).cpu()
        token = torch.multinomial(probabilities, 1, generator=generator).item()
        context.append(token)
        generated.append(token)
    model.train(was_training)
    return prompt + tokenizer.decode(generated)


def train(args):
    validate_train_args(args)
    out = Path(args.out_dir)
    if not args.resume and any(
        (out / name).exists() for name in ("config.json", "metrics.jsonl", "checkpoint.pt")
    ):
        raise ValueError(
            "output directory already has a run; use --resume or choose a new directory"
        )
    with stop_at_step_boundary() as stop:
        tokenizer, train_data, val_data, fingerprint = load_data(args.data, args.block_size)
        model_config = ModelConfig(
            len(tokenizer.characters), args.block_size, args.n_layer, args.n_head, args.n_embd
        )
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
                raise ValueError("--steps cannot be smaller than the checkpoint's completed step")
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
                    train_data, args.batch_size, args.block_size, batch_generator, args.device
                )
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, y)
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
                        "tokens_seen": step * args.batch_size * args.block_size,
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
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
        "eval_interval",
        "log_interval",
        "eval_batches",
        "checkpoint_interval",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if not 0 <= args.seed < 2**63 - 2:
        raise ValueError("--seed must be between 0 and 2**63 - 3")
    if args.n_embd % args.n_head:
        raise ValueError("n_embd must be divisible by n_head")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    trainer = commands.add_parser(
        "train", help="train or resume; --steps is total completed updates"
    )
    trainer.add_argument("--data", required=True, help="plain UTF-8 corpus path")
    trainer.add_argument(
        "--out-dir", required=True, help="artifact directory (metrics append on resume)"
    )
    for name, default in (
        ("steps", 100),
        ("batch-size", 1),
        ("block-size", ModelConfig.block_size),
        ("n-layer", ModelConfig.n_layer),
        ("n-head", ModelConfig.n_head),
        ("n-embd", ModelConfig.n_embd),
        ("seed", 1337),
        ("eval-interval", 10),
        ("log-interval", 1),
        ("eval-batches", 2),
        ("checkpoint-interval", 10),
    ):
        trainer.add_argument(f"--{name}", type=int, default=default)
    trainer.add_argument("--lr", type=float, default=3e-4)
    trainer.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    trainer.add_argument("--resume", help="trusted checkpoint; identical corpus/settings required")
    generator = commands.add_parser("generate", help="sample from a trusted checkpoint on CPU")
    generator.add_argument("--checkpoint", required=True)
    generator.add_argument("--prompt", required=True)
    generator.add_argument("--max-new-tokens", type=int, default=100)
    commands.add_parser(
        "self-test", help="bounded CPU checks using tiny models and temporary files"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "train":
            train(args)
        elif args.command == "generate":
            if args.max_new_tokens < 0:
                raise ValueError("--max-new-tokens must be nonnegative")
            checkpoint = load_checkpoint(args.checkpoint)
            model = GPT(ModelConfig(**checkpoint["config"]["model"]))
            model.load_state_dict(checkpoint["model"])
            tokenizer = CharacterTokenizer(checkpoint["tokenizer"]["characters"])
            print(
                sample(
                    model,
                    tokenizer,
                    args.prompt,
                    args.max_new_tokens,
                    checkpoint["config"]["training"]["seed"] + 2,
                )
            )
        else:
            from model_checks import self_test

            self_test()
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
