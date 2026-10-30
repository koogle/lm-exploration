"""Bounded single-GPU RL on math and code problems with checkable answers.

Answers come from a trajectory source, an automatic checker has rewarded them, and an
algorithm turns rewarded answers into updates. The package keeps those parts apart:

- `settings.py`: what a recipe sets, in sections (`algorithm`, `trajectories`,
  `rollouts`, `optimizer`, plus the shared `replay_*` fields).
- `algorithms.py`: the algorithms by name. `on_policy/` holds those that train only on
  answers the current weights just sampled; `off_policy/` those that also train on older
  or other models' answers, and the off-policy trajectory sources.
- `rollouts.py`: sampling answers with the current weights and rewarding them.
- `batch.py`, `policy_gradient.py`: what the algorithms share.
- `loop.py`: the training loop, checkpoints and provenance (`loop.train`).
"""

from trainer.rl.settings import Settings

__all__ = ["Settings"]
