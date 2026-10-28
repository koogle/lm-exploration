"""Pretraining replay that any post-training loop attaches (SFT, RL).

Replay trains on windows of pretraining text next to the task, to limit forgetting. The
windows come from a raw pretraining snapshot through the pretraining reader
(`training_data.raw_token_stream.RawTokenMixture`), with replay's own source weights:
consecutive `BOS ids EOS` documents of one source, every target but BOS trained.

A trainer's settings inherit `ReplaySettings` and call `check_replay()`;
`"replay": true` switches replay on, and `ReplayHook.attach` returns None when it is off.
A trainer backpropagates its own loss scaled by `1 - replay_weight`, then calls
`backward(model)`, which adds `replay_weight` × the next-token loss of the next replay
windows to the same gradients before the optimizer step. The trainer saves `state()`
with its checkpoints and records `identity()` and the `SOURCES` it hashes and copies
into the run. Validation is the loss on each source's first validation windows.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from training_data import raw_token_stream
from training_data.raw_token_stream import RawTokenMixture
from training_data.subword import LOSS_IGNORE_INDEX


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


def summed_loss(model, x, y):
    """(summed next-token cross-entropy, target count) of windows, BF16 autocast."""
    import torch
    from torch.nn import functional as F

    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        logits = model(x)
    value = F.cross_entropy(
        logits.float().flatten(0, 1), y.flatten(), ignore_index=LOSS_IGNORE_INDEX, reduction="sum"
    )
    return value, (y != LOSS_IGNORE_INDEX).sum()


class ReplayHook:
    # Source files a run hashes and copies, so it records the replay code it used.
    SOURCES: ClassVar[dict] = {
        "raw_token_stream": raw_token_stream.__file__,
        "replay_hook": __file__,
    }

    def __init__(self, settings, data_root, max_context):
        if settings.replay_context > max_context:
            raise ValueError("Replay context exceeds the model's context")
        manifest = Path(data_root) / settings.replay_snapshot / "manifest.json"
        context = settings.replay_context
        self.batch_size, self.weight = settings.replay_batch_size, settings.replay_weight

        # The snapshot is the parent's pretraining data, which its pretraining run already
        # admitted (with `allow_unreviewed_data` while audits are pending).
        def reader(split, weights):
            return RawTokenMixture(manifest, context, split, allow_unreviewed=True, weights=weights)

        self.data = reader("train", settings.replay_sources)
        self.windows = {
            source: reader("validation", {source: 1}).next_batch(settings.replay_validation_windows)
            for source in sorted(settings.replay_sources)
        }

    @classmethod
    def attach(cls, settings, data_root, max_context):
        """A hook for `settings`, or None when replay is off."""
        if not settings.replay:
            return None
        return cls(settings, data_root, max_context)

    def identity(self):
        return self.data.identity

    def state(self):
        return self.data.state_dict()

    def load_state(self, state):
        self.data.load_state_dict(state)

    def backward(self, model):
        """Add `replay_weight` × the mean next-token loss of the next `replay_batch_size`
        windows to the model's gradients; returns that loss."""
        import torch

        device = next(model.parameters()).device
        value, count = summed_loss(model, *self.data.next_batch(self.batch_size, device))
        loss = value / count
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite replay loss")
        (self.weight * loss).backward()
        return loss.item()

    def validation(self, model):
        """{source: mean next-token loss} over the fixed validation windows."""
        import torch

        device = next(model.parameters()).device
        results = {}
        with torch.inference_mode():
            for source, (x, y) in self.windows.items():
                value, count = summed_loss(model, x.to(device), y.to(device))
                results[source] = (value / count).item()
        return results
