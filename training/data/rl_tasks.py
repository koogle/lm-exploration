"""Prompts with checkable answers for RL (`rl-pool-v1`), and their rewards.

A pool row is one problem: `id`, `task`, `split` (`train` or `validation`), `source`,
`license` and `question`, plus for `math` the gold final answer `answer`, and for
`code` the function `declaration` and unit `tests`, run in the sandbox's `test_style`
(`functions` unless the row names `pytest`). The prompt is the SFT prompt of the
task, so an SFT model answers in the trained format:

- math: `Question: ...\\nAnswer:`; the completion is correct if Math-Verify finds its
  final answer equal to `answer`.
- code: `code_prompt`; the completion is the program, correct if every test passes in
  the code sandbox (`training_data.sandbox`).

A completion that does not end in one of its family's stop tokens within the token
limit scores 0 regardless. The family's `answer_format` says where the answer is:

- `plain`: the whole completion (the subword SFT formats above).
- `boxed`: for math the last `\\boxed{...}`, compared with `answer` by Math-Verify (LaTeX
  and expressions); for code the last fenced Python block. Prompts asking for these are
  the family's (`model_hybrid.post_train`).
"""

import hashlib
import json
import re
from pathlib import Path

from training_data.code_sft import code_prompt, mbpp_row, opencodeinstruct_row
from training_data.math_sft import answer_matches, ngrams, question_prompt

FORMAT = "rl-pool-v1"
# (repo, revision) of the optional sources.
MATH_LIGHTEVAL = ("DigitalLearningGmbH/MATH-lighteval", "0530c78699ea5e8eb5530600900e1f328b48acad")
KODCODE = ("KodCode/KodCode-Light-RL-10K", "dcf78a8bbba9a613b596ce993c4921a38687dfcc")
TASKS = ("math", "code")
ANSWER_FORMATS = ("plain", "boxed")
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def prompt(row):
    if row["task"] == "math":
        return question_prompt(row["question"])
    return code_prompt(row["question"], row["declaration"])


def last_boxed(text):
    """Contents of the last \\boxed{...} (balanced braces), or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for index in range(start + len("\\boxed{") - 1, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len("\\boxed{") : index]
    return None


def boxed_matches(text, gold):
    """Whether the last \\boxed{} of `text` equals `gold` (Math-Verify, LaTeX or expression)."""
    from math_verify import parse, verify

    boxed = last_boxed(text)
    if boxed is None:
        return False
    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${boxed}$")))
    except Exception:  # noqa: BLE001 - an unparsable answer earns nothing
        return False


def program(text, answer_format):
    """The code a completion submits: all of it (`plain`) or its last Python block."""
    if answer_format == "plain":
        return text
    blocks = CODE_BLOCK.findall(text)
    return blocks[-1] if blocks else None


def rewards(rows, completions, finished, sandbox, answer_format):
    """1.0 or 0.0 per (row, completion text, ended in a stop token), reading answers in
    `answer_format`; code runs in `sandbox`."""
    if answer_format not in ANSWER_FORMATS:
        raise ValueError(f"answer_format must be one of {ANSWER_FORMATS}")
    scores = [0.0] * len(rows)
    code = []
    for index, (row, text, done) in enumerate(zip(rows, completions, finished, strict=True)):
        if not done:
            continue
        if row["task"] == "math":
            if answer_format == "plain":
                scores[index] = float(answer_matches(text, row["answer"])[1])
            else:
                scores[index] = float(boxed_matches(text, row["answer"]))
        elif (source := program(text, answer_format)) is not None:
            code.append((index, (source, row["tests"], row.get("test_style", "functions"))))
    for (index, _), result in zip(code, sandbox.run([pair for _, pair in code]), strict=True):
        scores[index] = float(result["passed"])
    return scores


def is_heldout(text, fraction):
    """A stable hash-selected `fraction` of problems, by their text."""
    value = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    return value / 2**64 < fraction


def math_lighteval_row(row, index, levels, heldout_fraction):
    """A pool row of a MATH (lighteval) training problem of `levels`, or None."""
    answer = last_boxed(row["solution"]) if row["level"] in levels else None
    if not answer:
        return None
    return {
        "id": f"math-train-{index}",
        "task": "math",
        "split": "validation" if is_heldout(row["problem"], heldout_fraction) else "train",
        "question": row["problem"],
        "answer": answer,
        "source": f"{MATH_LIGHTEVAL[0]}@{MATH_LIGHTEVAL[1]} train row {index}",
        "license": "MIT",
    }


def kodcode_row(row, difficulties, heldout_fraction):
    """A pool row of a KodCode-Light-RL problem with one function and pytest tests, or None."""
    if row["gpt_difficulty"] not in difficulties or len(row["test_info"]) != 1:
        return None
    declaration = row["test_info"][0]["function_declaration"].strip()
    if not declaration.startswith(("def ", "class ")):
        declaration = f"def {declaration}"
    if not declaration.endswith(":"):
        declaration += ":"
    return {
        "id": f"kodcode-{row['question_id']}",
        "task": "code",
        "split": "validation" if is_heldout(row["question"], heldout_fraction) else "train",
        "question": row["question"],
        "declaration": declaration,
        "tests": row["test"],
        "test_style": "pytest",
        "solution": row["solution"],
        "source": f"{KODCODE[0]}@{KODCODE[1]} {row['question_id']}",
        "license": "CC-BY-NC-4.0",
    }


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
    math_lighteval=(),
    kodcode=(),
    check=None,
    math_levels=("Level 1", "Level 2", "Level 3"),
    kodcode_difficulties=("easy", "medium"),
    heldout_fraction=0.02,
):
    """Write `pool.jsonl` and `report.json` to `output`.

    Math: GSM8K training problems. The first 500 are the SFT validation problems
    (`math_sft.build`); the first `math_validation` of them are pool validation, the
    other 500 − `math_validation` are left out, and the rest are training. Code:
    the first `code_train` usable OpenCodeInstruct problems of `code_sft.RL_SHARD` (a
    shard code SFT does not use) are training and MBPP's validation split is validation. Rows sharing a 13-word n-gram with a benchmark
    problem (`math_tests`, `code_tests`) are dropped.

    Optional sources: MATH training problems of `math_levels` (`math_lighteval` rows; the
    answer is the solution's last `\\boxed{}`) and KodCode-Light-RL problems of
    `kodcode_difficulties` with one function (`kodcode` rows; pytest tests), each with a
    hash-selected `heldout_fraction` as validation. A KodCode problem is kept only if its
    reference solution passes its tests (`check`: the code sandbox's `run`).
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
    for index, row in enumerate(math_lighteval):
        keep(math_lighteval_row(row, index, math_levels, heldout_fraction))
    kodcode = list(kodcode)
    candidates = [kodcode_row(row, kodcode_difficulties, heldout_fraction) for row in kodcode]
    candidates = [row for row in candidates if row is not None]
    counts["unusable"] += len(kodcode) - len(candidates)
    if candidates:
        results = check([(r["solution"], r["tests"], r["test_style"]) for r in candidates])
        counts["reference_failed"] = sum(not r["passed"] for r in results)
        for row, result in zip(candidates, results, strict=True):
            if result["passed"]:
                keep({k: v for k, v in row.items() if k != "solution"})
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
