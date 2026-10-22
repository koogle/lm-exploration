"""Validated, framework-independent settings for subword DDP training."""

import math
from collections.abc import Mapping
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
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    gradient_clip_norm: float = 1.0
    data_loader: str = "token_shards"
    validation_windows_per_source: int = 1
    log_every: int = 10
    seed: int = 42
    checkpoint_every: int = 100
    evaluate_every: int = 100
    pilot: bool = False
    allow_unreviewed_data: bool = False
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
            "validation_windows_per_source",
            "log_every",
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
        if type(self.allow_unreviewed_data) is not bool:
            raise ValueError("allow_unreviewed_data must be boolean")
        if type(self.prefetch_batches) is not int or not 0 <= self.prefetch_batches <= 64:
            raise ValueError("prefetch_batches must be an integer from 0 to 64")
        if self.data_loader not in ("token_shards", "raw", "live"):
            raise ValueError("data_loader must be token_shards, raw or live")
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
        for name in (
            "muon_lr",
            "muon_momentum",
            "peak_lr",
            "min_lr_ratio",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "gradient_clip_norm",
        ):
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
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

    def as_dict(self):
        return asdict(self)

    def learning_rate(self, tokens):
        if tokens < self.warmup_tokens:
            return self.peak_lr * min((tokens + 1) / self.warmup_tokens, 1.0)
        progress = min((tokens - self.warmup_tokens) / (self.target_tokens - self.warmup_tokens), 1)
        return self.peak_lr * (
            self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        )


@dataclass(frozen=True)
class DistributedRuntime:
    """Per-process launch details, separate from the shared training recipe."""

    rank: int
    local_rank: int
    world_size: int
    master_addr: str
    master_port: int

    def __post_init__(self):
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("Runtime world_size must be a positive integer")
        for name in ("rank", "local_rank"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < self.world_size:
                raise ValueError(f"Runtime {name} must be within world_size")
        if self.rank != self.local_rank:
            raise ValueError("This trainer supports one host; rank must equal local_rank")
        if (
            not isinstance(self.master_addr, str)
            or not self.master_addr
            or any(c.isspace() or c in "/?#@" for c in self.master_addr)
        ):
            raise ValueError("Runtime master_addr must be a hostname or IP address")
        if type(self.master_port) is not int or not 1 <= self.master_port <= 65535:
            raise ValueError("Runtime master_port must be between 1 and 65535")

    @classmethod
    def from_torchrun(cls, environment: Mapping[str, str]):
        """Resolve torchrun's environment once at the command-line boundary."""
        try:
            return cls(
                rank=int(environment["RANK"]),
                local_rank=int(environment["LOCAL_RANK"]),
                world_size=int(environment["WORLD_SIZE"]),
                master_addr=environment["MASTER_ADDR"],
                master_port=int(environment["MASTER_PORT"]),
            )
        except KeyError as exc:
            raise ValueError(f"Missing torchrun launch variable: {exc.args[0]}") from exc

    @property
    def init_method(self):
        # Brackets distinguish an IPv6 address from the port separator.
        host = self.master_addr
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"tcp://{host}:{self.master_port}"


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
