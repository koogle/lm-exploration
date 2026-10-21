"""Explicit run policies, separate from reusable training settings."""

from .config import DistributedConfig


def validate_small_5b_run(config: DistributedConfig) -> None:
    """Validate the dedicated three-H100 modern 9.85M / 5B workflow."""
    if (config.n_layer, config.n_head, config.n_embd) != (8, 3, 192):
        raise ValueError("The small-5B workflow requires the modern 9.85M architecture")
    if config.world_size != 3:
        raise ValueError("The small-5B workflow allocates exactly three GPUs")
    if config.pilot or config.target_tokens > 5_000_000_000:
        raise ValueError(
            "The small-5B workflow requires a non-pilot target no larger than 5B tokens"
        )
