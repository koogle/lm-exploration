"""Validated, framework-independent settings for subword DDP training."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DistributedConfig:
    world_size: int = 3
    global_windows: int = 48
    microbatch: int = 16
    crop_padding: bool = False
    loss_chunk_size: int = 0
    compile_model: bool = False
    cache_rope: bool = False
    context: int = 2048
    target_tokens: int = 20_000_000_000
    warmup_tokens: int = 20_000_000
    optimizer: str = "muon"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
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
        for name in ("crop_padding", "compile_model", "cache_rope"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if type(self.loss_chunk_size) is not int or self.loss_chunk_size < 0:
            raise ValueError("loss_chunk_size must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a 32-bit unsigned integer")
        if type(self.pilot) is not bool:
            raise ValueError("pilot must be boolean")
        if type(self.experimental_data) is not bool:
            raise ValueError("experimental_data must be boolean")
        if self.experimental_data and (
            self.pilot
            or (self.n_layer, self.n_head, self.n_embd) not in ((8, 3, 192), (14, 12, 768))
            or self.target_tokens > (10_000_000_000 if self.n_embd == 768 else 5_000_000_000)
        ):
            raise ValueError(
                "Experimental data is scoped to the authorized modern 9.85M / 5B or 124M / 10B runs"
            )
        if type(self.prefetch_batches) is not int or not 0 <= self.prefetch_batches <= 64:
            raise ValueError("prefetch_batches must be an integer from 0 to 64")
        if self.world_size not in (1, 2, 3, 8):
            raise ValueError("Supported GPU counts: 1, 2, 3, 8")
        if self.global_windows % (self.world_size * self.microbatch):
            raise ValueError("Global windows must divide into whole per-GPU microbatches")
        if self.n_embd % self.n_head or (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE requires an even integral head dimension")
        if self.warmup_tokens >= self.target_tokens:
            raise ValueError("Warmup must end before the token target")
        if self.optimizer not in ("adamw", "muon"):
            raise ValueError("optimizer must be adamw or muon")
        for name in ("peak_lr", "min_lr_ratio", "weight_decay", "muon_lr", "muon_momentum"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid {name}")
        if self.peak_lr <= 0 or not 0 <= self.min_lr_ratio <= 1 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.muon_lr <= 0 or not 0 <= self.muon_momentum < 1:
            raise ValueError("Invalid Muon settings")
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
