"""Answers sampled by the weights being trained, and their rewards.

`Scheduler` draws training prompts per task and rests prompts that carry no signal.
`collect` samples answers with the model family's sampler (`trainer/families.py`) and
scores them (`training_data.rl_tasks`, in each problem's answer format): 1 for an answer
that ends in one of the family's stop tokens and is correct (Math-Verify for math, every
unit test passing in the code sandbox for code), else 0.
`Sampler` is the trajectory source of on-policy training: every step it samples a group
of answers to each drawn prompt with the current weights.
"""

from concurrent.futures import ThreadPoolExecutor

from training_data import rl_tasks

from trainer.rl.batch import Batch
from trainer.rl.ranks import Ranks


class Scheduler:
    """Draws training prompts per task; a prompt solved by none or all of its group on
    two visits in a row rests for `rest` steps."""

    def __init__(self, rows, rest, rng):
        self.rows, self.rest, self.rng = rows, rest, rng
        self.state = {}

    def draw(self, task, count, step):
        rows = [r for r in self.rows if r["task"] == task]
        ready = [r for r in rows if self.state.get(r["id"], {}).get("until", 0) <= step]
        return self.rng.sample(ready if len(ready) >= count else rows, count)

    def update(self, row, mean, step):
        state = self.state.setdefault(row["id"], {"until": 0, "extreme": None, "streak": 0})
        if mean in (0.0, 1.0):
            state["streak"] = state["streak"] + 1 if state["extreme"] == mean else 1
            state["extreme"] = mean
            if state["streak"] >= 2:
                state.update(until=step + self.rest, streak=0, extreme=None)
        else:
            state.update(streak=0, extreme=None)

    def resting(self, step):
        return sum(s["until"] > step for s in self.state.values())


def collect(family, model, rows, rollouts, temperature, generator, tokenizer, sandbox, top_k=0):
    """A `Batch` for `rows`, sampled by `family` at `temperature` (0 is greedy), one
    sampler batch per task, recording the sampler's `top_k` likeliest tokens; code answers
    are checked in a background thread while math is sampled."""
    completions, finished = [None] * len(rows), [None] * len(rows)
    stops = [None] * len(rows)
    sampler = [None] * len(rows)
    rewards = [0.0] * len(rows)
    model.eval()
    with ThreadPoolExecutor(max_workers=1) as checker:
        pending = None
        for task in ("code", "math"):
            indices = [i for i, r in enumerate(rows) if r["task"] == task]
            if not indices:
                continue
            samples, ended = family.sample(
                model,
                [rows[i]["prompt_ids"] for i in indices],
                rollouts.token_limit(task),
                temperature,
                generator,
                top_k,
                rollouts.engine,
            )
            for j, (i, c, d) in enumerate(
                zip(indices, samples.completions, samples.finished, strict=True)
            ):
                completions[i], finished[i], stops[i] = c, d, ended[j]
                if samples.logprobs is not None:
                    n = len(c) + d
                    sampler[i] = (
                        samples.logprobs[j, :n],
                        None if samples.top_ids is None else samples.top_ids[j, :n],
                        None if samples.top_ids is None else samples.top_logprobs[j, :n],
                    )
            chosen = [rows[i] for i in indices]
            texts = [tokenizer.decode(c, skip_special_tokens=True) for c in samples.completions]
            scores = (rl_tasks.rewards, chosen, texts, samples.finished, sandbox)
            if task == "code":
                pending = (indices, checker.submit(*scores))
            else:
                for i, r in zip(indices, scores[0](*scores[1:]), strict=True):
                    rewards[i] = r
        if pending:
            for i, r in zip(pending[0], pending[1].result(), strict=True):
                rewards[i] = r
    model.train()
    return Batch(rows, completions, finished, rewards, stops, sampler=sampler)


class Sampler:
    """Every step: `prompts_per_step` training prompts, `math_share` of them math, and
    `samples_per_prompt` answers to each from the current weights, trained in one update.
    Every rank draws the same prompts and samples its share of them."""

    updates_per_batch = 1

    def __init__(self, rollouts, scheduler, sample, ranks=None):
        self.rollouts, self.scheduler, self.sample = rollouts, scheduler, sample
        self.ranks = ranks or Ranks()

    def batch(self, step):
        rollouts, group = self.rollouts, self.rollouts.samples_per_prompt
        math_prompts = round(rollouts.prompts_per_step * rollouts.math_share)
        drawn = self.scheduler.draw("math", math_prompts, step) + self.scheduler.draw(
            "code", rollouts.prompts_per_step - math_prompts, step
        )
        mine = self.ranks.shard(drawn)
        batch = self.sample([row for row in mine for _ in range(group)])
        means = [
            sum(batch.rewards[index * group : (index + 1) * group]) / group
            for index in range(len(mine))
        ]
        # Every rank updates the scheduler with every prompt's mean, so they stay equal.
        for row, mean in zip(drawn, self.ranks.unshard(means), strict=True):
            self.scheduler.update(row, mean, step)
        batch.group = group
        return batch

    def statistics(self, step):
        return {"resting_prompts": self.scheduler.resting(step)}
