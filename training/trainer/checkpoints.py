"""Historical format-1 checkpoint loading, compatibility checks and provenance.

The active trainer owns its durable DDP checkpoint save/restore in loop.py.
These helpers retain published character-checkpoint interpretation.
"""

import hashlib
import platform
import subprocess
from pathlib import Path

import torch


def load_checkpoint(path):
    # Only load trusted checkpoints: the format contains Python RNG objects.
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
        # Matching dimensions alone do not identify the architecture/optimizer layout.
        and old["model_name"] == config["model_name"]
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


def runtime_provenance(device, model_source, model_definition=None, *, model_dependencies=None):
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
        "data": root / "data" / "loading.py",
        "dependencies": root / "uv.lock",
    }
    # A scaled model imports its modules from another file; hash that lineage too.
    if model_definition and Path(model_definition) != Path(model_source):
        files["model_definition"] = Path(model_definition)
    # Variants can inherit both a scaled configuration and the original block definition.
    files.update({name: Path(path) for name, path in (model_dependencies or {}).items()})
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
