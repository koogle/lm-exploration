"""Group-relative policy optimization (arXiv:2402.03300), on-policy."""

from dataclasses import dataclass

from trainer.rl import policy_gradient


@dataclass(frozen=True)
class GRPO:
    """Advantage: reward minus the mean reward of the prompt's group, divided by the
    group's standard deviation only with `scale_by_std` (off by default, as in Dr. GRPO,
    arXiv:2503.20783). Loss: the policy gradient −A·∇log p on every answer token; the
    answers come from the current weights, so no importance ratio is needed."""

    name: str = "on_policy/grpo"
    scale_by_std: bool = False

    SOURCES = ("sampler",)
    needs_behaviour = False
    skips_zero_advantage = True

    def advantages(self, batch):
        return policy_gradient.group_advantages(batch.rewards, batch.group, self.scale_by_std)

    def loss(self, tokens):
        return policy_gradient.loss(tokens)
