"""Training settings. Runs come from recipes (../recipes/) and train on a Modal GPU."""

import math
from dataclasses import dataclass

# Training runs only on Modal GPU workers; there is no CPU or MPS training path.
DEVICE = "cuda"


@dataclass
class TrainConfig:
    data: str
    steps: int = 100
    batch_size: int = 1
    lr: float = 3e-4
    seed: int = 1337
    eval_interval: int = 10
    log_interval: int = 1
    eval_batches: int = 2
    checkpoint_interval: int = 10
    resume: str | None = None


def validate_train_args(args):
    # Settings only; the CUDA check happens when a worker starts training.
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
