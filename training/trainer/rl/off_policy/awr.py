"""Advantage-weighted regression (arXiv:1910.00177) on answers from any source."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AWR:
    """Likelihood training of every answer, weighted by how much better than average it
    scored; no importance ratio, so the answers may come from another model.

    Advantage: reward minus the mean reward of the batch's answers to the same task (no
    value model; answers need not come in groups). Each answer's tokens are trained with
    weight w = min(exp(A / beta), max_weight): −w·∇log p. A small `beta` approaches
    training on the best answers only; a large one, on all answers equally.
    """

    name: str = "off_policy/awr"
    beta: float = 1.0
    max_weight: float = 20.0

    SOURCES = ("reuse", "file")
    needs_behaviour = False
    skips_zero_advantage = False

    def __post_init__(self):
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be positive")
        if not math.isfinite(self.max_weight) or self.max_weight < 1:
            raise ValueError("max_weight must be at least 1")

    def advantages(self, batch):
        means = {}
        for task in {row["task"] for row in batch.rows}:
            rewards = [
                r for r, row in zip(batch.rewards, batch.rows, strict=True) if row["task"] == task
            ]
            means[task] = sum(rewards) / len(rewards)
        return [r - means[row["task"]] for r, row in zip(batch.rewards, batch.rows, strict=True)]

    def loss(self, tokens):
        weight = (tokens.advantage / self.beta).exp().clamp(max=self.max_weight)
        token_weight = weight[:, None] * tokens.mask
        loss = -(tokens.scale[:, None] * token_weight * tokens.logp).sum()
        return loss, {"awr_weight": token_weight.sum().item()}
