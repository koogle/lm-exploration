"""Run setup, evaluation and the core optimizer-step loop. No run starts on import."""

import json
import random
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from training_data.loading import get_batch, load_data

from .checkpoints import (
    load_checkpoint,
    restore_checkpoint,
    runtime_provenance,
    save_checkpoint,
    validate_resume,
)
from .config import DEVICE, validate_train_args
from .sampling import sample


@contextmanager
def stop_signal_handler():
    requested = [False]

    def request_stop(signum, frame):
        requested[0] = True  # Never raise while optimizer weights are being updated.

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield requested
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@torch.no_grad()
def evaluate(model, data, args):
    # Fixed held-out windows, independent of the training batch RNG.
    generator = torch.Generator().manual_seed(args.seed + 1)
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        tokens, targets = get_batch(
            data, args.batch_size, model.config.block_size, generator, DEVICE
        )
        logits = model(tokens)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(args, data, run_dir, *, architecture_module, model_config, resume=None, max_seconds=None):
    # Called by trainer/recipe.py on a Modal GPU worker; see ../recipes/README.md.
    selected_architecture = architecture_module
    model_class = selected_architecture.GPT
    if not torch.cuda.is_available():
        raise ValueError("training runs on Modal GPU workers; CUDA is not available")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    validate_train_args(args)
    out = Path(run_dir)
    if not resume and any(
        (out / name).exists() for name in ("config.json", "metrics.jsonl", "checkpoints")
    ):
        raise ValueError("run directory already has a run; resume it or choose a new directory")
    with stop_signal_handler() as stop:
        tokenizer, train_data, val_data, fingerprint = load_data(data, model_config.block_size)
        if model_config.vocab_size != len(tokenizer.characters):
            raise ValueError("model vocabulary must match the dataset tokenizer")
        settings = {
            name: getattr(args, name)
            for name in (
                "steps",
                "batch_size",
                "lr",
                "seed",
                "eval_interval",
                "log_interval",
                "eval_batches",
                "checkpoint_interval",
            )
        }
        settings.update(
            device=DEVICE,
            dtype="float32",
            optimizer="AdamW",
            betas=[0.9, 0.95],
            weight_decay=0.1,
            gradient_clip_norm=1.0,
        )
        config = {
            "model_name": selected_architecture.__package__,
            "model": asdict(model_config),
            "training": settings,
            "tokenizer": tokenizer.config(),
            "dataset": {"path": str(Path(data).resolve()), **fingerprint},
            "runtime": runtime_provenance(
                DEVICE,
                selected_architecture.__file__,
                sys.modules[model_class.__module__].__file__,
            ),
        }
        checkpoint = load_checkpoint(resume) if resume else None
        if checkpoint:
            validate_resume(checkpoint, config, fingerprint, args.steps)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        batch_generator = torch.Generator().manual_seed(args.seed)
        model = model_class(model_config).to(device=DEVICE, dtype=torch.float32)
        config["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        step, elapsed_before, last_loss = 0, 0.0, None
        if checkpoint:
            step, elapsed_before, last_loss = restore_checkpoint(
                checkpoint, model, optimizer, batch_generator
            )
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Model: {config['parameter_count']:,} parameters; device: {DEVICE}; FP32",
            flush=True,
        )
        started = time.perf_counter()
        saved_step = step

        def save():
            # A new file per save: an interrupted write never damages an earlier checkpoint.
            nonlocal saved_step
            save_checkpoint(
                out / "checkpoints" / f"step-{step:06d}.pt",
                config=config,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                step=step,
                batch_generator=batch_generator,
                fingerprint=fingerprint,
                elapsed_seconds=elapsed_before + time.perf_counter() - started,
                train_loss=last_loss,
            )
            saved_step = step

        # Progress bar only in an interactive shell; Modal logs get plain JSON lines.
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
                    DEVICE,
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
                if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                    stop[0] = True
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
        if step != saved_step:
            save()
        latest = out / "checkpoints" / f"step-{step:06d}.pt"
        text = sample(model, tokenizer, "", 80, args.seed + 2)
        (out / "sample.txt").write_text(text, encoding="utf-8")
        print(
            f"{'Stopped' if stop[0] else 'Completed'} at step {step}; "
            f"{config['parameter_count']} parameters; checkpoint: {latest}",
            flush=True,
        )
