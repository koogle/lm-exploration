"""Trusted checkpoint I/O, resume compatibility, RNG restoration and run provenance.

Format version 1 is unchanged: existing checkpoints remain readable.
"""

import hashlib
import platform
import random
import subprocess
from pathlib import Path

import torch


def rng_state(batch_generator):
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "batches": batch_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, batch_generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    batch_generator.set_state(state["batches"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def load_checkpoint(path):
    # Only load trusted checkpoints: this restores Python RNG objects as well.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    return checkpoint


def validate_resume(checkpoint, config, fingerprint, steps):
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
    if not compatible or any(old["training"][key] != config["training"][key] for key in fixed):
        raise ValueError(
            "resume requires identical data, tokenizer, model and optimizer/batch/seed/device settings"
        )
    if steps < checkpoint["step"]:
        raise ValueError("steps cannot be smaller than the checkpoint's completed step")


def restore_checkpoint(checkpoint, model, optimizer, batch_generator):
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint["rng"], batch_generator)
    return tuple(checkpoint[key] for key in ("step", "elapsed_seconds", "train_loss"))


def save_checkpoint(
    path,
    *,
    config,
    model,
    tokenizer,
    optimizer,
    step,
    batch_generator,
    fingerprint,
    elapsed_seconds,
    train_loss,
):
    state = {
        "format_version": 1,
        "config": config,
        "model": model.state_dict(),
        "tokenizer": tokenizer.config(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rng": rng_state(batch_generator),
        "dataset_fingerprint": fingerprint,
        "elapsed_seconds": elapsed_seconds,
        "train_loss": train_loss,
    }
    torch.save(state, path)


def runtime_provenance(device, model_source):
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True, timeout=2
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    files = {
        "model": Path(model_source),
        "training": root / "trainer" / "loop.py",
        "training_config": root / "trainer" / "config.py",
        "checkpoints": root / "trainer" / "checkpoints.py",
        "sampling": root / "trainer" / "sampling.py",
        "cli": root / "cli.py",
        "data": root / "data" / "loading.py",
        "dependencies": root / "uv.lock",
    }
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "device": device,
        "git_commit": commit,
        "source_hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in files.items()
            if path.is_file()
        },
        "cuda": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
    }
