"""Training settings; model dimensions belong in 100m/model.py."""

import math
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    # All normal training choices are here, not a long command line.
    data: str = str(ROOT / "data" / "raw" / "tinyshakespeare.txt")
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
