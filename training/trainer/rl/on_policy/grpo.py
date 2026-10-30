"""Group-relative policy optimization (arXiv:2402.03300), on-policy."""

from dataclasses import dataclass

from trainer.rl import policy_gradient


@dataclass(frozen=True)
class GRPO:
    """Advantage: reward minus the mean reward of the prompt's group, divided by the
    group's standard deviation only with `scale_by_std` (off by default, as in Dr. GRPO,
    arXiv:2503.20783). Loss: the policy gradient −A·∇log p on every answer token; the
    answers come from the current weights, so no importance ratio is needed.

    `score`: `"plain"` pushes up ∇log p; `"centered"` replaces it by Score Centering's
    ∇log p − E_q[∇log p] over the sampler's distribution q, estimated from the sampler's
    `centering_top_k` likeliest tokens (`scores.py`). Centering needs answers sampled
    from the plain policy (temperature 1)."""

    name: str = "on_policy/grpo"
    scale_by_std: bool = False
    score: str = "plain"
    centering_top_k: int = 32

    SOURCES = ("sampler",)
    needs_behaviour = False
    skips_zero_advantage = True

    def __post_init__(self):
        policy_gradient.check_score(self)

    @property
    def sampler_top_k(self):
        """Top tokens per answer token the sampler records for this algorithm."""
        return self.centering_top_k if self.score == "centered" else 0

    def advantages(self, batch):
        return policy_gradient.group_advantages(batch.rewards, batch.group, self.scale_by_std)

    def loss(self, tokens):
        return policy_gradient.loss(tokens, score=self.score)
