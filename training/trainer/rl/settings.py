"""RL settings, in the sections a recipe writes under `"settings"`:

    {
      "steps": 300,
      "rollouts": {"prompts_per_step": 64, "samples_per_prompt": 8, "math_share": 0.5},
      "algorithm": {"name": "grpo", "policy": "on"},
      "optimizer": {"lr": 1e-6, "warmup_steps": 20},
      "replay": true, "replay_snapshot": "...", ...
    }

Omitted fields take the defaults below; `Settings.from_dict` builds and checks them.
The `replay_*` fields are the ones SFT shares (`trainer/replay_hook.py`).
"""

import math
from dataclasses import dataclass, field

from trainer.replay_hook import ReplaySettings
from trainer.rl import algorithms


def positive_int(owner, *names):
    for name in names:
        value = getattr(owner, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Rollouts:
    """Prompts drawn per step and how their answers are sampled.

    A prompt whose group is all wrong or all right on two visits in a row rests for
    `rest_steps` steps.
    """

    prompts_per_step: int = 64
    samples_per_prompt: int = 8
    math_share: float = 0.5
    math_tokens: int = 256
    code_tokens: int = 384
    temperature: float = 1.0
    rest_steps: int = 50
    # Sampler engine (`trainer/rollout.py`): "graph" replays custom-kernel CUDA graphs.
    engine: str = "graph"

    def __post_init__(self):
        positive_int(self, "prompts_per_step", "samples_per_prompt", "math_tokens", "code_tokens")
        if self.samples_per_prompt < 2:
            raise ValueError("Group advantages need at least two samples per prompt")
        if not 0 <= self.math_share <= 1:
            raise ValueError("math_share must be in [0, 1]")
        if not 0 < self.temperature <= 2:
            raise ValueError("temperature must be in (0, 2]")
        if type(self.rest_steps) is not int or self.rest_steps < 0:
            raise ValueError("rest_steps must be a nonnegative integer")
        if self.engine not in ("eager", "graph"):
            raise ValueError("engine must be eager or graph")

    def token_limit(self, task):
        return self.math_tokens if task == "math" else self.code_tokens


@dataclass(frozen=True)
class Optimizer:
    """AdamW; the learning rate warms up linearly over `warmup_steps` updates."""

    lr: float = 1e-6
    warmup_steps: int = 20
    weight_decay: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.lr) or not 0 < self.lr <= 1e-4:
            raise ValueError("Invalid learning rate")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        if self.weight_decay < 0:
            raise ValueError("Invalid weight_decay")

    def learning_rate(self, update):
        """The rate for optimizer update `update` (0-based)."""
        return self.lr * min(1.0, (update + 1) / max(1, self.warmup_steps))


@dataclass(frozen=True)
class Settings(ReplaySettings):
    # Rollout batches; each takes the algorithm's `updates_per_batch` optimizer updates.
    steps: int = 300
    rollouts: Rollouts = field(default_factory=Rollouts)
    algorithm: algorithms.GRPO = field(default_factory=algorithms.GRPO)
    optimizer: Optimizer = field(default_factory=Optimizer)
    # KL(policy ‖ parent), k3 estimate on answer tokens; always measured, added at kl_coef.
    kl_coef: float = 0.0
    # KL(sampler ‖ learner), k3 estimate on the sampled tokens, added at sampler_kl_coef.
    sampler_kl_coef: float = 0.0
    # Answers per forward and backward pass.
    micro_batch: int = 16
    seed: int = 42
    evaluate_every: int = 50
    save_every: int = 100
    max_seconds: int = 72000

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        return cls(
            rollouts=Rollouts(**values.pop("rollouts", {})),
            algorithm=algorithms.make(values.pop("algorithm", {"name": "grpo"})),
            optimizer=Optimizer(**values.pop("optimizer", {})),
            **values,
        )

    def __post_init__(self):
        positive_int(self, "steps", "micro_batch", "evaluate_every", "save_every", "max_seconds")
        for coef in (self.kl_coef, self.sampler_kl_coef):
            if not math.isfinite(coef) or coef < 0:
                raise ValueError("Invalid KL coefficient")
        if self.algorithm.sampler_top_k and self.rollouts.temperature != 1:
            raise ValueError("Score centering needs sampling at temperature 1")
        if self.optimizer.warmup_steps >= self.steps * self.algorithm.updates_per_batch:
            raise ValueError("Warmup must end before the last update")
        if self.max_seconds > 20 * 60 * 60:
            raise ValueError("RL capped at 20 hours per invocation")
        self.check_replay()
