"""Prompts with checkable answers for RL (`rl-pool-v1`), and their rewards.

A pool row is one problem: `id`, `task`, `split` (`train` or `validation`), `source`,
`license` and `question`, plus for `math` the gold final answer `answer`, and for
`code` the function `declaration` and unit `tests`. The prompt is the SFT prompt of the
task, so an SFT model answers in the trained format:

- math: `Question: ...\\nAnswer:`; the completion is correct if Math-Verify finds its
  final answer equal to `answer`.
- code: `code_prompt`; the completion is the program, correct if every test passes in
  the code sandbox (`training_data.sandbox`).

A completion that does not end in EOS within the token limit scores 0 regardless.
"""

import hashlib
import json
from pathlib import Path

from training_data.code_sft import code_prompt, mbpp_row, opencodeinstruct_row
from training_data.math_sft import answer_matches, ngrams, question_prompt

FORMAT = "rl-pool-v1"
TASKS = ("math", "code")


def prompt(row):
    if row["task"] == "math":
        return question_prompt(row["question"])
    return code_prompt(row["question"], row["declaration"])


def rewards(rows, completions, finished, sandbox):
    """1.0 or 0.0 per (row, completion text, ended in EOS); code runs in `sandbox`."""
    scores = [0.0] * len(rows)
    code = []
    for index, (row, text, done) in enumerate(zip(rows, completions, finished, strict=True)):
        if not done:
            continue
        if row["task"] == "math":
            scores[index] = float(answer_matches(text, row["answer"])[1])
        else:
            code.append((index, (text, row["tests"])))
    for (index, _), result in zip(code, sandbox.run([pair for _, pair in code]), strict=True):
        scores[index] = float(result["passed"])
    return scores


def pool_row(sft):
    """The pool row of a `code-sft-v1` row (or None)."""
    if sft is None:
        return None
    keys = ("id", "split", "question", "declaration", "tests", "source", "license")
    return {"task": "code", **{key: sft[key] for key in keys}}


def load(path):
    """Pool rows by split, and the file's SHA256."""
    raw = Path(path).read_bytes()
    pool = {"train": [], "validation": []}
    for line in raw.decode().splitlines():
        row = json.loads(line)
        if row["task"] not in TASKS or row["split"] not in pool:
            raise ValueError(f"Invalid pool row {row.get('id')}")
        pool[row["split"]].append(row)
    if not all(pool[split] for split in pool):
        raise ValueError("Pool needs train and validation rows")
    return pool, hashlib.sha256(raw).hexdigest()


def build(
    gsm8k_train,
    opencodeinstruct,
    mbpp_validation,
    math_tests,
    code_tests,
    output,
    math_validation=200,
    code_train=10000,
):
    """Write `pool.jsonl` and `report.json` to `output`.

    Math: GSM8K training problems. The first 500 are the SFT validation problems
    (`math_sft.build`); the first `math_validation` of them are pool validation, the
    other 500 − `math_validation` are left out, and the rest are training. Code:
    the first `code_train` usable OpenCodeInstruct problems of `code_sft.RL_SHARD` (a
    shard code SFT does not use) are training and MBPP's validation split is validation. Rows sharing a 13-word n-gram with a benchmark
    problem (`math_tests`, `code_tests`) are dropped.
    """
    output = Path(output)
    blocked = {
        "math": set().union(*(ngrams(q) for q in math_tests)),
        "code": set().union(*(ngrams(q) for q in code_tests)),
    }
    rows, counts = [], {"contaminated": 0, "unusable": 0}

    def keep(row):
        if row is None:
            counts["unusable"] += 1
        elif ngrams(row["question"]) & blocked[row["task"]]:
            counts["contaminated"] += 1
        else:
            rows.append(row)
            key = f"{row['task']}/{row['split']}"
            counts[key] = counts.get(key, 0) + 1

    for index, row in enumerate(gsm8k_train):
        if math_validation <= index < 500:
            continue
        keep(
            {
                "id": f"gsm8k-train-{index}",
                "task": "math",
                "split": "validation" if index < math_validation else "train",
                "question": row["question"],
                "answer": row["answer"].rsplit("####", 1)[1].strip(),
                "source": f"openai/gsm8k main/train row {index}",
                "license": "MIT",
            }
        )
    for row in opencodeinstruct:
        if counts.get("code/train", 0) >= code_train:
            break
        keep(pool_row(opencodeinstruct_row(row, "train")))
    for row in mbpp_validation:
        keep(pool_row(mbpp_row(row, "validation")))
    output.mkdir(parents=True, exist_ok=False)
    path = output / "pool.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {
        "format": FORMAT,
        "pool_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decontamination": "13-word n-grams of GSM8K/SVAMP test and code benchmark problems",
        "kept": counts,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
