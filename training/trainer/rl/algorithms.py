"""The RL algorithms by name, as a recipe's `"algorithm": {"name": ..., **options}` gives
them. A name's folder says where its answers may come from: `on_policy/` algorithms
train only on answers the current weights just sampled; `off_policy/` algorithms also
train on older or other models' answers.

An algorithm is a frozen dataclass whose fields are its options. It declares the
trajectory `SOURCES` it accepts, whether it `needs_behaviour` log-probabilities (the
sampling weights' probabilities of each answer token, for importance ratios), and
whether answers with zero advantage are skipped. `advantages(batch)` gives one advantage
per answer of a `Batch`; `loss(tokens)` takes one micro-batch (`Tokens`) and returns the
scalar to minimize, whose gradient is what matters, and {statistic: sum over answer
tokens}.
"""

from trainer.rl.off_policy.awr import AWR
from trainer.rl.off_policy.ppo import PPO
from trainer.rl.on_policy.grpo import GRPO

ALGORITHMS = {algorithm.name: algorithm for algorithm in (GRPO(), PPO(), AWR())}


def make(spec):
    """The algorithm a recipe's `{"name": ..., **options}` describes."""
    spec = dict(spec)
    name = spec.get("name")
    if name not in ALGORITHMS:
        raise ValueError(f"algorithm name must be one of {sorted(ALGORITHMS)}")
    return type(ALGORITHMS[name])(**spec)
