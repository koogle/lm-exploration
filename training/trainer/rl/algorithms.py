"""RL algorithms: how rewards become advantages and a loss on the answer tokens.

An algorithm is a frozen dataclass with a `name` (its key in `ALGORITHMS` and in a
recipe's `"algorithm": {"name": ...}`) and a `policy`:

- `"on"`: one optimizer update per rollout batch, so every answer comes from the weights
  being updated.
- `"off"`: `updates_per_batch` updates per batch, each on its own share of the answers;
  after the first, answers come from older weights, so the algorithm must correct for
  that (`OFF_POLICY` algorithms only).

`advantages(rewards, group)` maps the rewards of consecutive groups of `group` answers
to the same prompt to one advantage per answer. `loss(tokens)` takes one micro-batch of
answers (`Tokens`) and returns the scalar to minimize, whose gradient is what matters,
and {statistic: sum over answer tokens}.
"""

from dataclasses import dataclass

import torch

from trainer.rl import scores

POLICIES = ("on", "off")


@dataclass
class Tokens:
    """One micro-batch of answers as an algorithm's loss sees it, `[batch, time]` unless
    noted: the learner's `logits` `[batch, time, vocab]` and its log-probabilities `logp`
    of the `targets` (both with gradient), `old_logp` of the weights that sampled the
    batch (detached; on-policy, `logp` itself), the answer-token `mask`, and per answer its
    `advantage` and constant normalization `scale` `[batch]` (see `loop.py`). When the
    algorithm asks the sampler to record its top tokens (`sampler_top_k`), the sampler's
    `sampler_top_ids` and `sampler_top_logprobs` `[batch, time, k]` at each target."""

    logits: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    logp: torch.Tensor
    old_logp: torch.Tensor
    advantage: torch.Tensor
    scale: torch.Tensor
    sampler_top_ids: torch.Tensor | None = None
    sampler_top_logprobs: torch.Tensor | None = None


@dataclass(frozen=True)
class GRPO:
    """Group-relative policy optimization (arXiv:2402.03300).

    Advantage: reward minus the mean reward of the prompt's group, divided by the
    group's standard deviation only with `scale_by_std` (off by default, as in Dr. GRPO,
    arXiv:2503.20783). Loss: PPO's clipped surrogate −min(r·A, clip(r, 1 ± clip)·A) per
    answer token, with r = p / p_old, written by its gradient −r·A·∇log p on tokens whose
    ratio is not clipped. On-policy p_old is p itself (detached), so r = 1 and the
    gradient is the plain policy gradient −A·∇log p.

    `score`: `"plain"` pushes up ∇log p as above; `"centered"` replaces it by Score
    Centering's ∇log p − E_q[∇log p] over the sampler's distribution q, estimated from the
    sampler's `centering_top_k` likeliest tokens (`scores.py`). Centering needs answers
    sampled from the plain policy (temperature 1).
    """

    name: str = "grpo"
    policy: str = "on"
    updates_per_batch: int = 1
    clip: float = 0.2
    scale_by_std: bool = False
    score: str = "plain"
    centering_top_k: int = 32

    def __post_init__(self):
        check_policy(self)
        if not 0 < self.clip < 1:
            raise ValueError("clip must be between 0 and 1")
        if self.score not in ("plain", "centered"):
            raise ValueError("score must be plain or centered")
        if type(self.centering_top_k) is not int or self.centering_top_k < 1:
            raise ValueError("centering_top_k must be a positive integer")

    @property
    def sampler_top_k(self):
        """Top tokens per answer token the sampler records for this algorithm."""
        return self.centering_top_k if self.score == "centered" else 0

    def advantages(self, rewards, group):
        result = []
        for start in range(0, len(rewards), group):
            chunk = rewards[start : start + group]
            mean = sum(chunk) / len(chunk)
            spread = (sum((r - mean) ** 2 for r in chunk) / (len(chunk) - 1)) ** 0.5
            divisor = spread + 1e-6 if self.scale_by_std else 1.0
            result += [(r - mean) / divisor for r in chunk]
        return result

    def loss(self, tokens):
        ratio = (tokens.logp - tokens.old_logp).detach().exp()
        advantage = tokens.advantage[:, None]
        # PPO zeroes the gradient where the ratio left the band in the advantage's direction.
        clipped = ((advantage > 0) & (ratio > 1 + self.clip)) | (
            (advantage < 0) & (ratio < 1 - self.clip)
        )
        weight = tokens.scale[:, None] * ratio * advantage * (~clipped & tokens.mask)
        score = tokens.logp
        if self.score == "centered":
            score = scores.centered(
                tokens.logits, tokens.targets, tokens.sampler_top_ids, tokens.sampler_top_logprobs
            )
        loss = -(weight * score).sum()
        return loss, {"clipped_tokens": (clipped & tokens.mask).sum().item()}


ALGORITHMS = {"grpo": GRPO}
# Algorithms that correct for answers sampled from older weights.
OFF_POLICY = {"grpo"}


def check_policy(algorithm):
    if algorithm.policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    updates = algorithm.updates_per_batch
    if type(updates) is not int or updates < 1:
        raise ValueError("updates_per_batch must be a positive integer")
    if algorithm.policy == "on" and updates != 1:
        raise ValueError("An on-policy algorithm takes one update per rollout batch")
    if algorithm.policy == "off" and (updates < 2 or algorithm.name not in OFF_POLICY):
        raise ValueError(
            f"Off-policy training needs updates_per_batch >= 2 and one of {sorted(OFF_POLICY)}"
        )


def make(spec):
    """The algorithm a recipe's `{"name": ..., **options}` describes."""
    spec = dict(spec)
    name = spec.get("name")
    if name not in ALGORITHMS:
        raise ValueError(f"algorithm name must be one of {sorted(ALGORITHMS)}")
    return ALGORITHMS[name](**spec)
