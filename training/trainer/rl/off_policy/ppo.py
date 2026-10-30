"""PPO's clipped surrogate (arXiv:1707.06347) with group-relative advantages, on answers
the model sampled itself and reuses for several updates."""

from dataclasses import dataclass

from trainer.rl import policy_gradient


@dataclass(frozen=True)
class PPO:
    """Advantage as in GRPO: reward minus the group mean (divided by the group's standard
    deviation with `scale_by_std`). Loss: −min(r·A, clip(r, 1 ± clip)·A) per answer token,
    with r = p / p_old and p_old the probabilities of the weights that sampled the batch,
    computed before its first update. `score` and `centering_top_k` as in GRPO."""

    name: str = "off_policy/ppo"
    clip: float = 0.2
    scale_by_std: bool = False
    score: str = "plain"
    centering_top_k: int = 32

    SOURCES = ("reuse",)
    needs_behaviour = True
    skips_zero_advantage = True

    def __post_init__(self):
        policy_gradient.check_score(self)
        if not 0 < self.clip < 1:
            raise ValueError("clip must be between 0 and 1")

    @property
    def sampler_top_k(self):
        """Top tokens per answer token the sampler records for this algorithm."""
        return self.centering_top_k if self.score == "centered" else 0

    def advantages(self, batch):
        return policy_gradient.group_advantages(batch.rewards, batch.group, self.scale_by_std)

    def loss(self, tokens):
        return policy_gradient.loss(tokens, self.clip, self.score)
