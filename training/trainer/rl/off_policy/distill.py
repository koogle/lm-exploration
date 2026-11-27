"""Distillation on the student's own answers (on-policy distillation, as in Qwen3,
arXiv:2505.09388 §4.5, and MobileLLM-R1.5): the teacher's next-token distribution at
every token of answers the student sampled."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Distill:
    """Minimizes reverse KL(student ‖ teacher) at every answer token, over the teacher's
    listed tokens (`training_data.distillation`). Rewards play no part: every answer
    trains, whether or not it is correct, because the teacher's distribution corrects each
    step of it.

    `temperature` below 1 sharpens the teacher: its listed log-probabilities are divided
    by it and renormalized over the listed tokens, so the student learns the tokens the
    teacher would sample at that temperature instead of its full spread.
    """

    name: str = "off_policy/distill"
    temperature: float = 1.0

    SOURCES = ("scored",)
    needs_behaviour = False
    needs_teacher = True
    skips_zero_advantage = False

    def __post_init__(self):
        if not math.isfinite(self.temperature) or not 0 < self.temperature <= 1:
            raise ValueError("temperature must be in (0, 1]")

    def targets(self, logprobs):
        """The teacher's listed log-probabilities `[..., k]` as the loss uses them."""
        if self.temperature == 1:
            return logprobs
        return (logprobs / self.temperature).log_softmax(-1)

    def advantages(self, batch):
        return [0.0] * len(batch.rows)

    def loss(self, tokens):
        kl = tokens.teacher_kl * tokens.mask
        return (tokens.scale[:, None] * kl).sum(), {"teacher_kl": kl.detach().sum().item()}
