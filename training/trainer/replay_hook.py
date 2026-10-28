"""Pretraining replay that any post-training loop attaches (SFT, RL).

Replay trains on windows of pretraining text from a raw snapshot
(`training_data.replay`) next to the task, to limit forgetting. A trainer's settings
inherit `ReplaySettings` and call `check_replay()`; `"replay": true` switches replay on,
and `ReplayHook.attach` returns None when it is off. The hook owns the window sampling and its RNG state, the next-token loss
on every target except BOS, and the fixed validation windows. A trainer backpropagates
its own loss scaled by `1 - replay_weight`, then calls `backward(model)`, which adds
`replay_weight` × the replay loss to the same gradients before the optimizer step; it
records `identity()`, `state()` and the `SOURCES` it hashes and copies into the run.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from training_data import replay as replay_data
from training_data.subword import LOSS_IGNORE_INDEX

from trainer.batches import collate


@dataclass(frozen=True)
class ReplaySettings:
    # `replay` switches replay on; it then needs a raw snapshot under the data volume,
    # its source weights, windows per step and the replay loss weight.
    replay: bool = False
    replay_snapshot: str = ""
    replay_sources: dict = field(default_factory=dict)
    replay_batch_size: int = 0
    replay_context: int = 1024
    replay_weight: float = 0.0
    replay_validation_windows: int = 4

    def check_replay(self):
        if type(self.replay) is not bool:
            raise ValueError("replay must be true or false")
        if self.replay and not (
            self.replay_snapshot
            and self.replay_sources
            and all(type(w) in (int, float) and w > 0 for w in self.replay_sources.values())
            and type(self.replay_batch_size) is int
            and self.replay_batch_size > 0
            and type(self.replay_context) is int
            and self.replay_context > 0
            and math.isfinite(self.replay_weight)
            and 0 < self.replay_weight < 1
            and type(self.replay_validation_windows) is int
            and self.replay_validation_windows > 0
        ):
            raise ValueError(
                "Replay needs a snapshot, positive source weights, batch size and context, "
                "and a weight between 0 and 1"
            )


def summed_loss(model, rows):
    """(summed next-token cross-entropy, target count) of replay windows, BF16 autocast."""
    import torch
    from torch.nn import functional as F

    device = next(model.parameters()).device
    x, y = collate(rows, device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(x)
    value = F.cross_entropy(
        logits.float().flatten(0, 1), y.flatten(), ignore_index=LOSS_IGNORE_INDEX, reduction="sum"
    )
    return value, (y != LOSS_IGNORE_INDEX).sum()


class ReplayHook:
    # Source files a run hashes and copies, so it records the replay code it used.
    SOURCES: ClassVar[dict] = {"replay": replay_data.__file__, "replay_hook": __file__}

    def __init__(self, settings, data_root, tokenizer, max_context, seed):
        if settings.replay_context > max_context:
            raise ValueError("Replay context exceeds the model's context")
        self.batch_size = settings.replay_batch_size
        self.weight = settings.replay_weight
        self.data = replay_data.Replay(
            Path(data_root) / settings.replay_snapshot,
            settings.replay_sources,
            settings.replay_context,
            tokenizer,
            seed,
        )
        self.windows = self.data.validation(settings.replay_validation_windows)

    @classmethod
    def attach(cls, settings, data_root, tokenizer, max_context, seed):
        """A hook for `settings`, or None when replay is off."""
        if not settings.replay:
            return None
        return cls(settings, data_root, tokenizer, max_context, seed)

    def identity(self):
        return self.data.identity()

    def state(self):
        return self.data.rng.getstate()

    def load_state(self, state):
        self.data.rng.setstate(state)

    def backward(self, model):
        """Add `replay_weight` × the mean next-token loss of `replay_batch_size` new
        windows to the model's gradients; returns that loss."""
        import torch

        value, count = summed_loss(model, [self.data.sample() for _ in range(self.batch_size)])
        loss = value / count
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite replay loss")
        (self.weight * loss).backward()
        return loss.item()

    def validation(self, model):
        """{source: mean next-token loss} over the fixed validation windows."""
        import torch

        totals = {}
        with torch.inference_mode():
            for row in self.windows:
                value, count = summed_loss(model, [row])
                total = totals.setdefault(row["bucket"], [0.0, 0])
                total[0] += value.item()
                total[1] += count.item()
        return {name: value / count for name, (value, count) in sorted(totals.items())}
