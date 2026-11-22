"""Distillation on the student's own answers (on-policy distillation, as in Qwen3,
arXiv:2505.09388 §4.5, and MobileLLM-R1.5): the teacher's next-token distribution at
every token of answers the student sampled."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Distill:
    """Minimizes KL(teacher ‖ student) at every answer token, over the teacher's listed
    tokens (`training_data.distillation`). Rewards play no part: every answer trains,
    whether or not it is correct, because the teacher's distribution corrects each
    step of it.
    """

    name: str = "off_policy/distill"

    SOURCES = ("scored",)
    needs_behaviour = False
    needs_teacher = True
    skips_zero_advantage = False

    def advantages(self, batch):
        return [0.0] * len(batch.rows)

    def loss(self, tokens):
        kl = tokens.teacher_kl * tokens.mask
        return (tokens.scale[:, None] * kl).sum(), {"teacher_kl": kl.detach().sum().item()}
