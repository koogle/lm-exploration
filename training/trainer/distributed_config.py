"""Validated, framework-independent settings for subword DDP training."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DistributedConfig:
    world_size: int = 3
    global_windows: int = 48
    microbatch: int = 16
    context: int = 2048
    target_tokens: int = 20_000_000_000
    warmup_tokens: int = 20_000_000
    peak_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    seed: int = 42
    checkpoint_every: int = 100
    evaluate_every: int = 100
    pilot: bool = False
    experimental_data: bool = False
    prefetch_batches: int = 0
    n_layer: int = 14
    n_head: int = 12
    n_embd: int = 768

    def __post_init__(self):
        for name in (
            "world_size",
            "global_windows",
            "microbatch",
            "context",
            "target_tokens",
            "warmup_tokens",
            "checkpoint_every",
            "evaluate_every",
            "n_layer",
            "n_head",
            "n_embd",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a 32-bit unsigned integer")
        if type(self.pilot) is not bool:
            raise ValueError("pilot must be boolean")
        if type(self.experimental_data) is not bool:
            raise ValueError("experimental_data must be boolean")
        if self.experimental_data and (
            self.pilot
            or (self.n_layer, self.n_head, self.n_embd) != (8, 3, 192)
            or self.target_tokens > 5_000_000_000
        ):
            raise ValueError("Experimental data is scoped to the authorized modern 9.85M / 5B run")
        if type(self.prefetch_batches) is not int or not 0 <= self.prefetch_batches <= 64:
            raise ValueError("prefetch_batches must be an integer from 0 to 64")
        if self.world_size not in (1, 2, 3):
            raise ValueError("Supported GPU counts: 1, 2, 3")
        if self.global_windows % (self.world_size * self.microbatch):
            raise ValueError("Global windows must divide into whole per-GPU microbatches")
        if self.n_embd % self.n_head or (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE requires an even integral head dimension")
        if self.warmup_tokens >= self.target_tokens:
            raise ValueError("Warmup must end before the token target")
        for name in ("peak_lr", "min_lr_ratio", "weight_decay"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid {name}")
        if self.peak_lr <= 0 or not 0 <= self.min_lr_ratio <= 1 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.pilot and self.target_tokens > 5_000_000:
            raise ValueError("Pending-audit pilot override is limited to 5M tokens")
        if not self.pilot and (self.n_layer, self.n_head, self.n_embd) not in (
            (14, 12, 768),
            (8, 3, 192),
        ):
            raise ValueError("Substantial runs use the 124.4M or modern 9.85M architecture")

    def as_dict(self):
        return asdict(self)

    def learning_rate(self, tokens):
        if tokens < self.warmup_tokens:
            return self.peak_lr * min((tokens + 1) / self.warmup_tokens, 1.0)
        progress = min((tokens - self.warmup_tokens) / (self.target_tokens - self.warmup_tokens), 1)
        return self.peak_lr * (
            self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        )


def validate_resume(saved, config, identity, source_hashes, torch_version):
    if saved.get("format") != "scholarly-ddp-v1":
        raise ValueError(
            "Unsupported checkpoint format; pilot/character checkpoints cannot resume here"
        )
    if saved["config"] != config.as_dict():
        raise ValueError("Resume requires identical training settings and GPU count")
    if saved["data_identity"] != identity:
        raise ValueError("Checkpoint dataset/tokenizer identity mismatch")
    if saved["source_sha256"] != source_hashes or saved["torch_version"] != torch_version:
        raise ValueError("Resume requires identical training source and PyTorch version")
