"""What an RL step trains on: a `Batch` of answers, and `Tokens`, one micro-batch of them
as an algorithm's loss sees it."""

from dataclasses import dataclass

import torch
from training_data.subword import EOS_TOKEN_ID, LOSS_IGNORE_INDEX


@dataclass
class Batch:
    """Answers to `rows` (pool rows with `prompt_ids`): new token IDs without EOS, whether
    each ended in EOS, and its reward. `group` is the number of consecutive answers to
    the same prompt when they were sampled as groups, else None."""

    rows: list
    completions: list
    finished: list
    rewards: list
    group: int | None = None

    def sequences(self):
        """{x, y}: prompt + answer (+ EOS) and its targets; only answer tokens are targets."""
        result = []
        for row, completion, done in zip(self.rows, self.completions, self.finished, strict=True):
            answer = completion + ([EOS_TOKEN_ID] if done else [])
            tokens = row["prompt_ids"] + answer
            labels = [LOSS_IGNORE_INDEX] * len(row["prompt_ids"]) + answer
            result.append({"x": tokens[:-1], "y": labels[1:]})
        return result


@dataclass
class Tokens:
    """One micro-batch of answers, `[batch, time]` unless noted: the learner's `logits`
    `[batch, time, vocab]` and its log-probabilities `logp` of the `targets` (both with
    gradient), `old_logp` of the weights that sampled the answers (detached; `logp` itself
    when they are the current weights), the answer-token `mask`, and per answer its
    `advantage` and constant normalization `scale` `[batch]` (see `loop.py`)."""

    logits: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    logp: torch.Tensor
    old_logp: torch.Tensor
    advantage: torch.Tensor
    scale: torch.Tensor
