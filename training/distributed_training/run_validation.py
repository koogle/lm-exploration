"""Run policies applied at launch and execution, not recipe construction."""

from .config import DistributedConfig


def validate_run(config: DistributedConfig) -> None:
    """Retain the bounded pilot policy independently of data review status."""
    if config.pilot and config.target_tokens > 5_000_000:
        raise ValueError("Pilot runs are limited to 5M tokens")
