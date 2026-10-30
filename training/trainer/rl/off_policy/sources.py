"""Off-policy trajectory sources.

- `Reuse`: like on-policy sampling, but each sampled batch is split into
  `updates_per_batch` shares with one optimizer update each, so later shares come from
  weights one or more updates old.
- `TrajectoryFile`: answers someone else wrote, with their rewards
  (`training_data.rl_trajectories`), drawn `answers_per_step` at a time in a shuffled
  order that is reshuffled after each pass.
"""

from trainer.rl.batch import Batch
from trainer.rl.rollouts import Sampler


class Reuse(Sampler):
    def __init__(self, rollouts, scheduler, sample, updates_per_batch):
        super().__init__(rollouts, scheduler, sample)
        self.updates_per_batch = updates_per_batch


class TrajectoryFile:
    updates_per_batch = 1

    def __init__(self, rows, answers_per_step, rng):
        if len(rows) < answers_per_step:
            raise ValueError("Fewer trajectories than answers_per_step")
        self.rows, self.answers_per_step, self.rng = rows, answers_per_step, rng
        self.order, self.passes = [], 0

    def batch(self, step):
        if len(self.order) < self.answers_per_step:
            self.order = list(range(len(self.rows)))
            self.rng.shuffle(self.order)
            self.passes += 1
        chosen = [self.rows[i] for i in self.order[: self.answers_per_step]]
        del self.order[: self.answers_per_step]
        return Batch(
            chosen,
            [row["completion_ids"] for row in chosen],
            [True] * len(chosen),
            [row["reward"] for row in chosen],
        )

    def statistics(self, step):
        return {"trajectory_passes": self.passes}
