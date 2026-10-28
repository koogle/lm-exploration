"""Bounded single-GPU RL on math and code problems with checkable answers.

The policy writes its own answers, an automatic checker rewards correct ones, and an
algorithm turns rewarded answers into updates. The package keeps those parts apart:

- `settings.py`: what a recipe sets, in sections (`rollouts`, `algorithm`, `optimizer`,
  plus the shared `replay_*` fields); `python -m trainer.rl --recipe <file>` prints a
  recipe's settings with every default filled in.
- `rollouts.py`: which prompts are drawn, how answers are sampled and rewarded.
- `algorithms.py`: how rewards become advantages and a loss (`ALGORITHMS`, by name), and
  whether answers may come from older weights (`policy`: `"on"` or `"off"`).
- `loop.py`: the training loop, checkpoints and provenance (`loop.train`).
"""

from trainer.rl.settings import Settings

__all__ = ["Settings"]
