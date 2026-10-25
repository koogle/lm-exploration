"""Python function-writing rows (`code-sft-v1`) and the KodCode + MBPP builder.

What is trained (see `training_data.sft`): the model reads a problem statement and the
function to write (`code_prompt`) and learns to write the complete reference solution,
then EOS. The solutions are KodCode's (written by GPT-4o and checked against KodCode's
unit tests) and MBPP's (written by people). Each row also keeps its unit tests
(`tests`, importing the solution as the module `solution`), which SFT ignores and RL
runs (`training_data.code_execution`).
"""

import ast
import hashlib
import json
import re
import warnings
from pathlib import Path

from training_data.math_sft import ngrams, normalize
from training_data.sft import SFTDataset

KODCODE = ("KodCode/KodCode-V1", "97d81f0695ddd7c5c02bfe16436db9a17a4a21e5")
KODCODE_SHARDS = 15
KODCODE_RL = ("KodCode/KodCode-Light-RL-10K", "dcf78a8bbba9a613b596ce993c4921a38687dfcc")
MBPP = ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c")
MBPP_PLUS = ("evalplus/mbppplus", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077")
HUMANEVAL_PLUS = ("evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b")
# Longest reference solution and problem statement kept, in characters.
MAX_SOLUTION = 1500
MAX_QUESTION = 1500


def code_prompt(question, declaration):
    return f"Question: {question.strip()}\nWrite a Python function `{declaration}`.\nAnswer:\n"


def declaration(source, name):
    """`def name(args):` of the top-level function `name` in `source`, or None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return f"def {name}({ast.unparse(node.args)}):"
    return None


def usable_tests(tests):
    """Whether `tests` defines `test_*` functions a plain runner can call: no arguments
    (pytest fixtures) and no decorators (parametrization)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(tests)
    except SyntaxError:
        return False
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]
    return bool(found) and not any(n.args.args or n.args.vararg or n.decorator_list for n in found)


def assert_tests(asserts, setup=""):
    """MBPP-style assert lines as one `test_*` function over the solution module."""
    body = "".join(f"    {line.strip()}\n" for line in asserts)
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
        name = re.fullmatch(r"def (\w+)\(.*\):", row["declaration"], re.DOTALL)
        if not name or declaration(row["solution"], name[1]) is None:
            raise ValueError(f"Solution of {row['id']} does not define {row['declaration']}")


CODE = CodeSFT()


def kodcode_row(row, split):
    """A `code-sft-v1` row from a KodCode row, or None if it is unusable here."""
    if row["style"] != "instruct" or row.get("filter_reason") or len(row["test_info"]) != 1:
        return None
    if len(row["solution"]) > MAX_SOLUTION or len(row["question"]) > MAX_QUESTION:
        return None
    if not usable_tests(row["test"]):
        return None
    try:
        signature = declaration(row["solution"], row["test_info"][0]["function_name"])
    except SyntaxError:
        return None
    if signature is None:
        return None
    # IDs end in the style (`_I` instruct, `_C` complete); a group is one problem.
    group = re.sub(r"_[A-Z]$", "", row["question_id"])
    return {
        "id": f"kodcode-{row['question_id']}",
        "group": f"kodcode-{group}",
        "split": split,
        "bucket": f"kodcode/{row['subset'].lower()}",
        "question": row["question"],
        "declaration": signature,
        "solution": row["solution"].strip(),
        "tests": row["test"],
        "source": f"{KODCODE[0]}@{KODCODE[1]} {row['question_id']}",
        "license": "CC-BY-NC-4.0",
    }


def mbpp_row(row, split):
    names = re.findall(r"(\w+)\s*\(", row["test_list"][0].removeprefix("assert"))
    signature = next((s for n in names if (s := declaration(row["code"], n))), None)
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


def build(kodcode_shards, mbpp_train, benchmarks, reserved_ids, output):
    """Write `examples.jsonl` and `report.json` (as `math_sft.build`) to `output`.

    `kodcode_shards` yields KodCode-V1 rows; KodCode IDs in `reserved_ids` (the RL pool)
    are left out. A row sharing a 13-word n-gram with a benchmark problem in
    `benchmarks` is dropped, as is a repeated problem statement. Two in 1,000 KodCode
    groups, chosen by hash, are validation; MBPP train rows are all training.
    """
    output = Path(output)
    blocked = set().union(*(ngrams(text) for text in benchmarks))
    counts = {"kodcode": 0, "mbpp": 0, "contaminated": 0, "reserved": 0, "unusable": 0}
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

    for shard in kodcode_shards:
        for row in shard:
            if row["question_id"] in reserved_ids:
                counts["reserved"] += 1
                continue
            group = re.sub(r"_[A-Z]$", "", row["question_id"])
            digest = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16)
            split = "validation" if digest % 1000 < 2 else "train"
            keep(kodcode_row(row, split), "kodcode")
    for row in mbpp_train:
        keep(mbpp_row(row, "train"), "mbpp")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "examples.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {
        "format": CodeSFT.FORMAT,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": {"kodcode": KODCODE, "mbpp": MBPP, "reserved": KODCODE_RL},
        "decontamination": "13-word n-grams of MBPP test/validation/prompt, MBPP+ and "
        "HumanEval+ problems",
        "splits": {
            split: sum(r["split"] == split for r in rows) for split in ("train", "validation")
        },
        "kept": counts,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
