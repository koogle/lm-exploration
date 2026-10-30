"""Off-policy trajectory sources.

- `Reuse`: like on-policy sampling, but each sampled batch is split into
  `updates_per_batch` shares with one optimizer update each, so later shares come from
  weights one or more updates old.
- `TrajectoryFile`: answers someone else wrote, with their rewards
  (`training_data.rl_trajectories`), drawn `answers_per_step` at a time in a shuffled
  order that is reshuffled after each pass.
"""

from trainer.rl.batch import Batch
from trainer.rl.ranks import Ranks
from trainer.rl.rollouts import Sampler


class Reuse(Sampler):
    def __init__(self, rollouts, scheduler, sample, updates_per_batch, ranks=None):
        super().__init__(rollouts, scheduler, sample, ranks)
        self.updates_per_batch = updates_per_batch


class TrajectoryFile:
    updates_per_batch = 1

    def __init__(self, rows, answers_per_step, rng, stop, ranks=None):
        """`stop`: the token that ends each (finished) written answer. Every rank draws the
        same answers and trains its share."""
        if len(rows) < answers_per_step:
            raise ValueError("Fewer trajectories than answers_per_step")
        self.rows, self.answers_per_step, self.rng = rows, answers_per_step, rng
        self.stop, self.ranks = stop, ranks or Ranks()
        self.order, self.passes = [], 0

    def batch(self, step):
        if len(self.order) < self.answers_per_step:
            self.order = list(range(len(self.rows)))
            self.rng.shuffle(self.order)
            self.passes += 1
        chosen = self.ranks.shard([self.rows[i] for i in self.order[: self.answers_per_step]])
        del self.order[: self.answers_per_step]
        return Batch(
            chosen,
            [row["completion_ids"] for row in chosen],
            [True] * len(chosen),
            [row["reward"] for row in chosen],
            [self.stop] * len(chosen),
        )

    def statistics(self, step):
        return {"trajectory_passes": self.passes}

    def state(self):
        return {"order": list(self.order), "passes": self.passes}

    def load_state(self, state):
        self.order, self.passes = list(state["order"]), state["passes"]
