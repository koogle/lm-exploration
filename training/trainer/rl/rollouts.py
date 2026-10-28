"""Prompts, sampled answers and their rewards.

`Scheduler` draws training prompts per task and rests prompts that carry no signal.
`collect` samples answers with the shared cached sampler (`trainer/rollout.py`) and
scores them (`training_data.rl_tasks`): 1 for an answer that ends in EOS and is correct
(Math-Verify for math, every unit test passing in the code sandbox for code), else 0.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from training_data import rl_tasks
from training_data.subword import EOS_TOKEN_ID, LOSS_IGNORE_INDEX

from trainer import rollout


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


@dataclass
class Batch:
    """Answers to `rows` (pool rows, a prompt's group consecutive): new token IDs
    without EOS, whether each ended in EOS, and its reward. When sampled (not greedy),
    `sampler[i]` holds the sampler's log-probabilities of answer i's tokens (EOS
    included) and, if recorded, its top tokens and their log-probabilities at each."""

    rows: list
    completions: list
    finished: list
    rewards: list
    sampler: list

    def sequences(self):
        """(inputs, targets) of prompt + answer (+ EOS); only answer tokens are targets."""
        result = []
        for row, completion, done in zip(self.rows, self.completions, self.finished, strict=True):
            answer = completion + ([EOS_TOKEN_ID] if done else [])
            tokens = row["prompt_ids"] + answer
            labels = [LOSS_IGNORE_INDEX] * len(row["prompt_ids"]) + answer
            result.append({"x": tokens[:-1], "y": labels[1:]})
        return result


def collect(model, rows, rollouts, temperature, generator, tokenizer, sandbox, top_k=0):
    """A `Batch` for `rows`, sampled at `temperature` (0 is greedy), one sampler batch per
    task, recording the sampler's `top_k` likeliest tokens; code answers are checked in a
    background thread while math is sampled."""
    completions, finished = [None] * len(rows), [None] * len(rows)
    sampler = [None] * len(rows)
    rewards = [0.0] * len(rows)
    model.eval()
    with ThreadPoolExecutor(max_workers=1) as checker:
        pending = None
        for task in ("code", "math"):
            indices = [i for i, r in enumerate(rows) if r["task"] == task]
            if not indices:
                continue
            samples = rollout.sample(
                model,
                [rows[i]["prompt_ids"] for i in indices],
                rollouts.token_limit(task),
                EOS_TOKEN_ID,
                temperature,
                generator,
                record_top_k=top_k,
            )
            for j, (i, c, d) in enumerate(
                zip(indices, samples.completions, samples.finished, strict=True)
            ):
                completions[i], finished[i] = c, d
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
    return Batch(rows, completions, finished, rewards, sampler)
