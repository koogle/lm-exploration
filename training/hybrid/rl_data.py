"""RL prompts with verifiable references, from a recipe (recipes/rl-*.json).

Each source is one pinned parquet file of math problems (reference: the final
answer, checked by `rewards.math_reward`) or code problems (reference: a pytest
file importing from `solution`, run by `rewards.code_reward`). `prepare` turns
every row into an OpenAI user message asking for a `\\boxed{}` answer or a single
Python block, drops prompts sharing a 13-gram with the pretraining benchmark
screen, holds out a hash-selected fraction, and writes `prompts.jsonl`,
`heldout.jsonl` and `summary.json` under `<root>/<recipe name>/`. `PromptStream`
serves training prompts in a seeded order, resumably.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .rewards import last_boxed

MATH_INSTRUCTION = (
    "Solve the problem step by step, then give the final answer as \\boxed{{answer}}."
)
CODE_INSTRUCTION = (
    "Write the Python function `{declaration}`. Give the complete solution, with any "
    "imports it needs, in one ```python code block."
)


def gsm8k_answer(text):
    return text.split("####")[-1].strip().replace(",", "")


def math_answer(solution):
    return last_boxed(solution)


def to_prompt(source, settings, row):
    """(user message, reference) for one row, or None when it has no usable reference."""
    kind = settings["kind"]
    if source == "gsm8k":
        problem, reference = row["question"], gsm8k_answer(row["answer"])
    elif kind == "math":
        if row["level"] not in settings["levels"]:
            return None
        problem, reference = row["problem"], math_answer(row["solution"])
    else:
        if row["gpt_difficulty"] not in settings["difficulties"] or len(row["test_info"]) != 1:
            return None
        declaration = row["test_info"][0]["function_declaration"].strip().rstrip(":")
        declaration = re.sub(r"^def\s+", "", declaration)
        instruction = CODE_INSTRUCTION.format(declaration=declaration)
        return f"{row['question'].strip()}\n\n{instruction}", row["test"]
    if not reference:
        return None
    return f"{problem.strip()}\n\n{MATH_INSTRUCTION.format()}", reference


def is_heldout(text, fraction):
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64 < fraction


def prepare(recipe, rows_by_source, root, tokenizer, screen=None):
    """Write the prompt files; returns the summary."""
    from .data import contaminated

    folder = Path(root) / recipe["name"]
    folder.mkdir(parents=True, exist_ok=True)
    train, heldout, counts = [], [], {}
    for source, rows in rows_by_source.items():
        settings = recipe["sources"][source]
        kept = dropped = screened = 0
        for index, row in enumerate(rows):
            prompt = to_prompt(source, settings, row)
            if prompt is None:
                dropped += 1
                continue
            text, reference = prompt
            if screen is not None:
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if contaminated(ids, screen, recipe["screen_ngram"]):
                    screened += 1
                    continue
            item = {
                "id": f"{source}-{index}",
                "source": source,
                "kind": settings["kind"],
                "messages": [{"role": "user", "content": text}],
                "reference": reference,
            }
            (heldout if is_heldout(text, recipe["heldout_fraction"]) else train).append(item)
            kept += 1
        counts[source] = {"kept": kept, "unusable": dropped, "contaminated": screened}
    for name, items in (("prompts", train), ("heldout", heldout)):
        with open(folder / f"{name}.jsonl", "w") as handle:
            for item in items:
                handle.write(json.dumps(item) + "\n")
    summary = {"recipe": recipe["name"], "train": len(train), "heldout": len(heldout)}
    summary["sources"] = counts
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle]


class PromptStream:
    """Training prompts in a seeded order, reshuffled each pass; the state is a count."""

    def __init__(self, prompts, seed=0):
        self.prompts, self.seed, self.position = prompts, seed, 0
        self._orders = {}

    def _order(self, epoch):
        if epoch not in self._orders:
            rng = np.random.default_rng([self.seed, epoch])
            self._orders = {epoch: rng.permutation(len(self.prompts))}
        return self._orders[epoch]

    def take(self, count):
        out = []
        for _ in range(count):
            epoch, index = divmod(self.position, len(self.prompts))
            out.append(self.prompts[self._order(epoch)[index]])
            self.position += 1
        return out

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]
