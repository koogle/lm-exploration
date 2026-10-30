"""PPO's clipped surrogate (arXiv:1707.06347) with group-relative advantages, on answers
the model sampled itself and reuses for several updates."""

from dataclasses import dataclass

from trainer.rl import policy_gradient


@dataclass(frozen=True)
class PPO:
    """Advantage as in GRPO: reward minus the group mean (divided by the group's standard
    deviation with `scale_by_std`). Loss: −min(r·A, clip(r, 1 ± clip)·A) per answer token,
    with r = p / p_old and p_old the probabilities of the weights that sampled the batch,
    computed before its first update."""

    name: str = "off_policy/ppo"
    clip: float = 0.2
    scale_by_std: bool = False

    SOURCES = ("reuse",)
    needs_behaviour = True
    skips_zero_advantage = True

    def __post_init__(self):
        if not 0 < self.clip < 1:
            raise ValueError("clip must be between 0 and 1")

    def advantages(self, batch):
        return policy_gradient.group_advantages(batch.rewards, batch.group, self.scale_by_std)

    def loss(self, tokens):
        return policy_gradient.loss(tokens, self.clip)
