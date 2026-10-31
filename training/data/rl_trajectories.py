"""Answers someone else wrote, with their rewards, for off-policy RL (`rl-trajectories-v1`).

One row per answer: the pool row of its problem (`training_data.rl_tasks`: `id`, `task`,
`split`, `question`, ...), the answer text `completion` as the policy would write it (for
code, the program), its `reward` (1.0 or 0.0, from the same checker as RL rewards) and
the `author` model. Several rows may answer the same problem. A trajectory set is
`/runs/<data-id>/trajectories.jsonl` with a `report.json` holding its SHA256.

`from_teacher` makes trajectories from the code teacher's answers
(`training_data.teacher`, `modal_scripts/modal_teacher.py`): every answer whose code
defines the problem's declared name, passing or not, so failures carry reward 0.
"""

import hashlib
import json
from pathlib import Path

from training_data import rl_tasks
from training_data.teacher import extract

FORMAT = "rl-trajectories-v1"
# Programs per sandbox request.
SANDBOX_BATCH = 2000


def load(path):
    """Training rows and the file's SHA256."""
    raw = Path(path).read_bytes()
    rows = []
    for line in raw.decode().splitlines():
        row = json.loads(line)
        if row["task"] not in rl_tasks.TASKS or row["reward"] not in (0.0, 1.0):
            raise ValueError(f"Invalid trajectory {row.get('id')}")
        if row["split"] == "train":
            # The answers are SFT-format texts: the problems keep their SFT prompts.
            rows.append(rl_tasks.styled(row, "sft"))
    if not rows:
        raise ValueError("Trajectories need training rows")
    return rows, hashlib.sha256(raw).hexdigest()


def from_teacher(sft_rows, answers, sandbox, author):
    """Trajectories of `code-sft-v1` rows from the teacher's answer texts per row."""
    rows, pairs = [], []
    for sft, texts in zip(sft_rows, answers, strict=True):
        if sft["split"] != "train":
            continue
        for text in texts:
            code = extract(text, sft)
            if code is not None:
                rows.append({**rl_tasks.pool_row(sft), "completion": code, "author": author})
                pairs.append((code, sft["tests"]))
    results = [
        result
        for start in range(0, len(pairs), SANDBOX_BATCH)
        for result in sandbox.run(pairs[start : start + SANDBOX_BATCH])
    ]
    for row, result in zip(rows, results, strict=True):
        row["reward"] = float(result["passed"])
    return rows
