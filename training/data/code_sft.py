"""Python function-writing rows (`code-sft-v1`) and the OpenCodeInstruct + MBPP builder.

What is trained (see `training_data.sft`): the model reads a problem statement and the
function or class to write (`code_prompt`) and learns to write the complete reference
solution, then EOS. The solutions are OpenCodeInstruct's and MBPP's (written by people).
OpenCodeInstruct (CC-BY-4.0) is teacher-generated: Qwen2.5-Coder-32B-Instruct
(Apache-2.0) wrote each solution and ten unit tests, and NVIDIA ran the tests; only
solutions that pass all ten are kept, without their example-usage code. Each row also
keeps its unit tests (`tests`, importing the solution as the module `solution`), which
SFT ignores and RL runs (`training_data.sandbox`).
"""

import ast
import hashlib
import json
import re
import textwrap
import warnings
from pathlib import Path

from training_data.math_sft import ngrams, normalize
from training_data.sft import SFTDataset

OPENCODEINSTRUCT = ("nvidia/OpenCodeInstruct", "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d")
OPENCODEINSTRUCT_SHARDS = 50
# Shards for SFT rows; the last shard is the RL pool's (`training_data.rl_tasks`).
SFT_SHARDS = range(6)
RL_SHARD = OPENCODEINSTRUCT_SHARDS - 1
MBPP = ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c")
MBPP_PLUS = ("evalplus/mbppplus", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077")
HUMANEVAL_PLUS = ("evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b")
# Longest reference solution and problem statement kept, in characters.
MAX_SOLUTION = 1500
MAX_QUESTION = 1500
# Top-level statements a solution keeps; example usage (calls, prints, assignments) goes.
KEPT = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def shard_name(index):
    return f"data/train-{index:05d}-of-{OPENCODEINSTRUCT_SHARDS:05d}.parquet"


def code_prompt(question, declaration):
    kind = "class" if declaration.startswith("class ") else "function"
    return f"Question: {question.strip()}\nWrite a Python {kind} `{declaration}`.\nAnswer:\n"


def parse(source):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def declaration(source, name):
    """`def name(args):` or `class name:` of top-level `name` in `source`, or None."""
    for node in parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return f"def {name}({ast.unparse(node.args)}):"
        if isinstance(node, ast.ClassDef) and node.name == name:
            return f"class {name}:"
    return None


def called(source, test):
    """The declaration of the first name `test` calls that `source` defines, or None."""
    names = re.findall(r"(\w+)\s*\(", test.strip().removeprefix("assert"))
    return next((s for n in names if (s := declaration(source, n))), None)


def definitions(source):
    """`source` with only its imports, functions and classes (comments inside kept)."""
    lines = source.splitlines()
    kept = []
    for node in parse(source).body:
        if isinstance(node, KEPT):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
            kept.append("\n".join(lines[start - 1 : node.end_lineno]))
    return "\n\n".join(kept)


def assert_tests(asserts, setup=""):
    """Assert statements as one `test_*` function over the solution module."""
    body = "".join(textwrap.indent(a.strip(), "    ") + "\n" for a in asserts)
    return f"from solution import *\n{setup.strip()}\n\ndef test_asserts():\n{body}"


class CodeSFT(SFTDataset):
    FORMAT = "code-sft-v1"
    FIELDS = ("bucket", "question", "declaration", "solution", "tests")

    def prompt(self, row):
        return code_prompt(row["question"], row["declaration"])

    def answer(self, row):
        return row["solution"].strip()

    def bucket(self, row):
        return row["bucket"]

    def fingerprints(self, row):
        return (normalize(row["question"]),)

    def check(self, row):
        name = re.fullmatch(r"(?:def|class) (\w+)\b.*:", row["declaration"], re.DOTALL)
        if not name or declaration(row["solution"], name[1]) is None:
            raise ValueError(f"Solution of {row['id']} does not define {row['declaration']}")


CODE = CodeSFT()


def opencodeinstruct_row(row, split):
    """A `code-sft-v1` row from an OpenCodeInstruct row whose solution passed all its
    tests, or None if it is unusable here."""
    statuses = json.loads(row["tests_execution_status"])
    if not statuses or any(s != "pass" for s in statuses):
        return None
    blocks = re.findall(r"```python\n(.*?)```", row["output"], re.DOTALL)
    if len(blocks) != 1 or len(row["input"]) > MAX_QUESTION:
        return None
    tests = json.loads(row["unit_tests"])
    try:
        solution = definitions(blocks[0]).strip()
        signature = called(solution, tests[0])
        parse(assert_tests(tests))
    except (SyntaxError, IndexError):
        return None
    if signature is None or len(solution) > MAX_SOLUTION:
        return None
    return {
        "id": f"oci-{row['id']}",
        "group": f"oci-{row['id']}",
        "split": split,
        "bucket": f"oci/{row['domain']}",
        "question": row["input"],
        "declaration": signature,
        "solution": solution,
        "tests": assert_tests(tests),
        "source": f"{OPENCODEINSTRUCT[0]}@{OPENCODEINSTRUCT[1]} {row['id']}",
        "license": "CC-BY-4.0",
    }


def mbpp_row(row, split):
    signature = called(row["code"], row["test_list"][0])
    if signature is None:
        return None
    return {
        "id": f"mbpp-{row['task_id']}",
        "group": f"mbpp-{row['task_id']}",
        "split": split,
        "bucket": "mbpp",
        "question": row["text"],
        "declaration": signature,
        "solution": row["code"].replace("\r\n", "\n").strip(),
        "tests": assert_tests(row["test_list"], row["test_setup_code"]),
        "source": f"{MBPP[0]}@{MBPP[1]} task {row['task_id']}",
        "license": "CC-BY-4.0",
    }


def benchmark_texts(mbpp_plus, humaneval_plus, mbpp_held_out):
    """Problem statements of the code benchmarks, for decontamination."""
    return [
        *(r["prompt"] for r in mbpp_plus),
        *(r["prompt"] for r in humaneval_plus),
        *(r["text"] for r in mbpp_held_out),
    ]


def build(shards, mbpp_train, benchmarks, output):
    """Write `examples.jsonl` and `report.json` (as `math_sft.build`) to `output`.

    `shards` yields the OpenCodeInstruct rows of `SFT_SHARDS`. A row sharing a 13-word
    n-gram with a benchmark problem in `benchmarks` is dropped, as is a repeated problem
    statement. Two in 1,000 OpenCodeInstruct problems, chosen by hash of their ID, are
    validation; MBPP train rows are all training.
    """
    output = Path(output)
    blocked = set().union(*(ngrams(text) for text in benchmarks))
    counts = {"opencodeinstruct": 0, "mbpp": 0, "contaminated": 0, "unusable": 0}
    rows, seen = [], set()

    def keep(row, name):
        if row is None:
            counts["unusable"] += 1
        elif ngrams(row["question"]) & blocked:
            counts["contaminated"] += 1
        elif normalize(row["question"]) in seen:
            counts["unusable"] += 1
        else:
            seen.add(normalize(row["question"]))
            rows.append(row)
            counts[name] += 1

    for shard in shards:
        for row in shard:
            digest = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
            split = "validation" if digest % 1000 < 2 else "train"
            keep(opencodeinstruct_row(row, split), "opencodeinstruct")
    for row in mbpp_train:
        keep(mbpp_row(row, "train"), "mbpp")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "examples.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {
        "format": CodeSFT.FORMAT,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": {
            "opencodeinstruct": [*OPENCODEINSTRUCT, [shard_name(i) for i in SFT_SHARDS]],
            "mbpp": MBPP,
        },
        "decontamination": "13-word n-grams of MBPP test/validation/prompt, MBPP+ and "
        "HumanEval+ problems",
        "splits": {
            split: sum(r["split"] == split for r in rows) for split in ("train", "validation")
        },
        "kept": counts,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
