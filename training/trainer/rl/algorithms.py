"""RL algorithms: how rewards become advantages and a loss on the answer tokens.

An algorithm is a frozen dataclass with a `name` (its key in `ALGORITHMS` and in a
recipe's `"algorithm": {"name": ...}`) and a `policy`:

- `"on"`: one optimizer update per rollout batch, so every answer comes from the weights
  being updated.
- `"off"`: `updates_per_batch` updates per batch, each on its own share of the answers;
  after the first, answers come from older weights, so the algorithm must correct for
  that (`OFF_POLICY` algorithms only).

`advantages(rewards, group)` maps the rewards of consecutive groups of `group` answers
to the same prompt to one advantage per answer. `loss(logp, old_logp, advantage, mask,
scale)` is the scalar to minimize for a micro-batch: `logp` are the learner's
log-probabilities of the answer tokens (with gradient), `old_logp` those of the weights
that the batch was sampled from (detached), `mask` marks answer tokens and `scale` is each
answer's constant normalization (see `loop.py`). It returns (loss, {statistic: sum}).
"""

from dataclasses import dataclass

import torch

POLICIES = ("on", "off")


@dataclass(frozen=True)
class GRPO:
    """Group-relative policy optimization (arXiv:2402.03300).

    Advantage: reward minus the mean reward of the prompt's group, divided by the
    group's standard deviation only with `scale_by_std` (off by default, as in Dr. GRPO,
    arXiv:2503.20783). Loss: the clipped surrogate −min(r·A, clip(r, 1 ± clip)·A) per
    answer token, with r = p / p_old. On-policy p_old is p itself (detached), so r = 1 and
    the gradient is the plain policy gradient −A·∇log p.
    """

    name: str = "grpo"
    policy: str = "on"
    updates_per_batch: int = 1
    clip: float = 0.2
    scale_by_std: bool = False

    def __post_init__(self):
        check_policy(self)
        if not 0 < self.clip < 1:
            raise ValueError("clip must be between 0 and 1")

    def advantages(self, rewards, group):
        result = []
        for start in range(0, len(rewards), group):
            chunk = rewards[start : start + group]
            mean = sum(chunk) / len(chunk)
            spread = (sum((r - mean) ** 2 for r in chunk) / (len(chunk) - 1)) ** 0.5
            divisor = spread + 1e-6 if self.scale_by_std else 1.0
            result += [(r - mean) / divisor for r in chunk]
        return result

    def loss(self, logp, old_logp, advantage, mask, scale):
        ratio = (logp - old_logp).exp()
        advantage = advantage[:, None]
        bounded = ratio.clamp(1 - self.clip, 1 + self.clip)
        surrogate = torch.minimum(ratio * advantage, bounded * advantage)
        loss = -(scale[:, None] * surrogate * mask).sum()
        clipped = ((ratio - bounded).abs() > 0) & mask
        return loss, {"clipped_tokens": clipped.sum().item()}


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
