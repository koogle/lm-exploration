"""Pretraining replay that any post-training loop attaches (SFT, RL).

Replay trains on windows of pretraining text from a raw snapshot
(`training_data.replay`) next to the task, to limit forgetting. A trainer's settings
inherit `ReplaySettings` and call `check_replay()`; `ReplayHook.attach` returns None when
replay is off. The hook owns the window sampling and its RNG state, the next-token loss
on every target except BOS, and the fixed validation windows; the trainer only decides
how `loss(model)` enters its objective and records `identity()`, `state()` and the
`SOURCES` it hashes and copies into the run.
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
    # A raw snapshot under the data volume, its source weights, and windows per step.
    replay_snapshot: str = ""
    replay_sources: dict = field(default_factory=dict)
    replay_batch_size: int = 0
    replay_context: int = 1024
    replay_weight: float = 0.0
    replay_validation_windows: int = 4

    def check_replay(self):
        enabled = (
            self.replay_snapshot,
            self.replay_sources,
            self.replay_batch_size,
            self.replay_weight,
        )
        if any(enabled) and not (
            all(enabled[:3])
            and type(self.replay_batch_size) is int
            and math.isfinite(self.replay_weight)
            and 0 < self.replay_weight < 1
            and all(type(w) in (int, float) and w > 0 for w in self.replay_sources.values())
            and type(self.replay_context) is int
            and self.replay_context > 0
            and type(self.replay_validation_windows) is int
            and self.replay_validation_windows > 0
        ):
            raise ValueError(
                "Replay needs a snapshot, positive source weights and batch size, "
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
        if not settings.replay_batch_size:
            return None
        return cls(settings, data_root, tokenizer, max_context, seed)

    def identity(self):
        return self.data.identity()

    def state(self):
        return self.data.rng.getstate()

    def load_state(self, state):
        self.data.rng.setstate(state)

    def loss(self, model):
        """Mean next-token loss of `replay_batch_size` new windows; differentiable."""
        value, count = summed_loss(model, [self.data.sample() for _ in range(self.batch_size)])
        return value / count

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
