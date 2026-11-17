"""Contest problems in LiveCodeBench's prompt, each answered with one verified program.

Each conversation is the harness's own zero-shot instruction for the problem
(`livecodebench.instruction`: its system text, the statement and the answer format)
and an assistant turn holding one fenced Python program, so that SFT teaches the
format, a single answer and a clean stop. A program is kept only if it passes the
problem's tests under the harness's grader (`livecodebench.grade`), run as the
harness runs it:

- `codecontests` (DeepMind CodeContests, CC-BY-4.0): stdin programs. Of the Python 3
  solutions, the shortest that passes the public, private and first generated tests
  (up to `MAX_TESTS`) is kept. Problems reading or writing files are skipped. Its
  Codeforces contests end in 2021-22.
- `leetcode` (LeetCodeDataset, Apache-2.0): `class Solution` methods under the starter
  code. Its `input_output` cases (`nums = [3,3], target = 6` -> `[0, 1]`) become the
  harness's functional tests (one JSON argument per line). The reference solution is
  kept as written if it passes; otherwise with `IMPORTS` written out at the top, if
  that passes, so every answer runs on its own. Problems dated from `CUTOFF` on, and
  cases whose values are not JSON literals (linked lists, trees), are skipped.
- `opencodereasoning2` (OpenCodeReasoning-2, CC-BY-4.0): DeepSeek-R1 reasoning traces,
  in thinking mode. The release leaves `question` blank and names the problem by
  (`dataset`, `split`, `index`) in TACO, APPS or CodeContests; `questions` builds that
  table from the pinned `QUESTION_SOURCES`. A trace is kept only if its program passed
  all of the release's tests (`pass_rate` 1.0) and its problem is not in a benchmark
  split (`BENCHMARK_SPLITS`). The answer after the reasoning is the release's program
  alone, in one fenced block, as the harness expects. open-r1/codeforces problems are
  skipped: the release does not pin that dataset's revision, so its indices may not
  match the current row order.
- `codeforces_cots` (open-r1/codeforces-cots, ODC-By): DeepSeek-R1 traces for
  Codeforces problems, statements inline, in thinking mode. Only finished generations
  of exact-output (`diff`) problems without interaction are used; the answer is the
  trace's last Python block, kept if it passes the row's public and private tests
  (up to `MAX_TESTS`). Its contests end in 2024-07; the release is decontaminated
  against LiveCodeBench, and preparation applies our screen as well.
"""

import ast
import json

from evaluation.benchmarks import livecodebench

CODECONTESTS = ("deepmind/code_contests", "802411c3010cb00d1b05bad57ca77365a3c699d6")
LEETCODE = ("newfacade/LeetCodeDataset", "215604aeed660029df7de2fea5a4d7b6ed476a08")
OPENCODEREASONING2 = ("nvidia/OpenCodeReasoning-2", "eadf535931451525f3e5621d0f960c240bc62fd9")
CODEFORCES_COTS = ("open-r1/codeforces-cots", "39ac85c150806230473c70ad72c31f6232fe3f41")
CUTOFF = "2025-01-01"
PYTHON3 = 3  # CodeContests `solutions.language`
MAX_TESTS = 30
MAX_CANDIDATES = 4
MAX_SOLUTION = 4000  # characters
# The star imports LeetCodeDataset's reference solutions are written against.
IMPORTS = (
    "from typing import *\nfrom functools import *\nfrom collections import *\n"
    "from itertools import *\nfrom heapq import *\nfrom bisect import *\n"
    "from math import *\nfrom string import *\nfrom operator import *\n"
)


def conversation(question_content, starter_code, program):
    problem = {"question_content": question_content, "starter_code": starter_code}
    return {
        "messages": [
            {"role": "user", "content": livecodebench.instruction(problem)},
            {"role": "assistant", "content": f"```python\n{program.strip()}\n```"},
        ],
        "tools": [],
    }


def _passes(problem, program):
    return livecodebench.grade(problem, f"```python\n{program}\n```") == 1.0


def codecontests_tests(row):
    """(input, output) tests: public, then private, then generated, up to MAX_TESTS."""
    tests = []
    for kind in ("public_tests", "private_tests", "generated_tests"):
        cases = row[kind] or {}
        tests += [
            {"input": i, "output": o}
            for i, o in zip(cases.get("input") or [], cases.get("output") or [], strict=True)
        ]
    return tests[:MAX_TESTS]


def codecontests(row):
    """A conversation from a CodeContests row, or None without a passing Python 3 program."""
    if row.get("input_file") or row.get("output_file"):
        return None
    tests = codecontests_tests(row)
    solutions = row["solutions"] or {}
    candidates = sorted(
        {
            code
            for language, code in zip(
                solutions.get("language") or [], solutions.get("solution") or [], strict=True
            )
            if language == PYTHON3 and len(code) <= MAX_SOLUTION
        },
        key=len,
    )
    if not tests or not candidates:
        return None
    problem = {"starter_code": "", "tests": tests}
    for program in candidates[:MAX_CANDIDATES]:
        if _passes(problem, program):
            return conversation(row["description"], "", program)
    return None


def _arguments(text):
    """`a = 1, b = [2]` -> the values as JSON lines, in order; None if not literals."""
    try:
        call = ast.parse(f"f({text})", mode="eval").body
        values = [ast.literal_eval(keyword.value) for keyword in call.keywords]
        values += [ast.literal_eval(arg) for arg in call.args]
        return "\n".join(json.dumps(v) for v in values)
    except (SyntaxError, ValueError, TypeError):
        return None


def _value(text):
    """An expected output as JSON. The dataset stores `str()` of the value, so text that
    is not a Python literal is a returned string (a string that reads as a literal
    then fails verification and is dropped)."""
    try:
        return json.dumps(ast.literal_eval(text))
    except (SyntaxError, ValueError, TypeError):
        return json.dumps(text)


def leetcode_tests(row):
    """Harness functional tests from a LeetCodeDataset row, or None if any case is not
    plain JSON values."""
    cases = row["input_output"]
    cases = ast.literal_eval(cases) if isinstance(cases, str) else cases
    tests = []
    for case in cases[:MAX_TESTS]:
        arguments, expected = _arguments(case["input"]), _value(case["output"])
        if arguments is None or expected is None:
            return None
        tests.append({"input": arguments, "output": expected})
    return tests or None


def leetcode(row):
    """A conversation from a LeetCodeDataset row, or None."""
    if str(row.get("estimated_date", "")) >= CUTOFF or len(row["completion"]) > MAX_SOLUTION:
        return None
    tests = leetcode_tests(row)
    if tests is None:
        return None
    problem = {"starter_code": row["starter_code"], "tests": tests}
    for program in (row["completion"], IMPORTS + row["completion"]):
        if _passes(problem, program):
            return conversation(row["problem_description"], row["starter_code"], program)
    return None


# (repo, revision, {split: files in row order}) for the problems OpenCodeReasoning-2
# indexes; an `index` counts rows across a split's files in this order.
QUESTION_SOURCES = {
    "taco": (
        "BAAI/TACO",
        "d593ed0a2becbbc952230bb89be09189bf1056dc",
        {"train": [f"ALL/train-{i:05d}-of-00009.parquet" for i in range(9)]},
    ),
    "apps": (
        "codeparrot/apps",
        "21e74ddf8de1a21436da12e3e653065c5213e9d1",
        {"train": ["train.jsonl"]},
    ),
    "code_contests": (
        CODECONTESTS[0],
        CODECONTESTS[1],
        {
            "train": [
                "data/train-00000-of-00039-e991a271dbfa9925.parquet",
                "data/train-00001-of-00039-e092fe56fda18715.parquet",
                "data/train-00002-of-00039-9cea23812e920e41.parquet",
                "data/train-00003-of-00039-e3822fccad6e083a.parquet",
                "data/train-00004-of-00039-cefe355b4667b27e.parquet",
                "data/train-00005-of-00039-b7580d2d846c2136.parquet",
                "data/train-00006-of-00039-65184bb9f7d61fde.parquet",
                "data/train-00007-of-00039-05785de21e8b8429.parquet",
                "data/train-00008-of-00039-7246e6b7423b404f.parquet",
                "data/train-00009-of-00039-b8c920f6629b57b2.parquet",
                "data/train-00010-of-00039-6de28ba20654f69b.parquet",
                "data/train-00011-of-00039-5de236be5188959d.parquet",
                "data/train-00012-of-00039-da9476a39a1bdbb7.parquet",
                "data/train-00013-of-00039-30b8c3829ee3b962.parquet",
                "data/train-00014-of-00039-dc3ebb07a3cba8e4.parquet",
                "data/train-00015-of-00039-19ccd7331d695677.parquet",
                "data/train-00016-of-00039-bf38b0908b322307.parquet",
                "data/train-00017-of-00039-ae5533a2f822e6ef.parquet",
                "data/train-00018-of-00039-8c793837880f5507.parquet",
                "data/train-00019-of-00039-d688fad5ee604390.parquet",
                "data/train-00020-of-00039-5d59387098675b73.parquet",
                "data/train-00021-of-00039-b257bf03d6876780.parquet",
                "data/train-00022-of-00039-1cfd39fa43c1917c.parquet",
                "data/train-00023-of-00039-d078bcb55e45cbf0.parquet",
                "data/train-00024-of-00039-f4e3da0e5661e6d1.parquet",
                "data/train-00025-of-00039-3f6ebfbaba5f4c70.parquet",
                "data/train-00026-of-00039-7d4898300894cbbe.parquet",
                "data/train-00027-of-00039-f8196766547533a2.parquet",
                "data/train-00028-of-00039-79a302af3c924863.parquet",
                "data/train-00029-of-00039-2b6615897d038115.parquet",
                "data/train-00030-of-00039-4135cc54050afc22.parquet",
                "data/train-00031-of-00039-40309dd907c042b7.parquet",
                "data/train-00032-of-00039-7b7d2068a3d9c359.parquet",
                "data/train-00033-of-00039-53b0f749aacff9c1.parquet",
                "data/train-00034-of-00039-a36ff0bff7d2a76f.parquet",
                "data/train-00035-of-00039-d28f9be60314601f.parquet",
                "data/train-00036-of-00039-146e1a11c054aeab.parquet",
                "data/train-00037-of-00039-995207c374a4e6f2.parquet",
                "data/train-00038-of-00039-96a59dd6a98cd075.parquet",
            ],
            "valid": ["data/valid-00000-of-00001-5e672c5751f060d3.parquet"],
        },
    ),
}
BENCHMARK_SPLITS = {("apps", "test"), ("taco", "test")}


def question_rows(dataset, rows):
    """{"question", "starter_code"} for each source row, in order."""
    if dataset == "code_contests":
        return [{"question": r["description"], "starter_code": ""} for r in rows]
    return [{"question": r["question"], "starter_code": r["starter_code"] or ""} for r in rows]


def split_think(generation):
    """(reasoning, answer) of a `<think>…</think>answer` generation, or None."""
    if not generation.startswith("<think>") or generation.count("</think>") != 1:
        return None
    reasoning, answer = generation[len("<think>") :].split("</think>")
    return reasoning.strip(), answer.strip()


def opencodereasoning2(row, questions):
    """A thinking-mode conversation from an OpenCodeReasoning-2 row, or None.
    `questions` maps "dataset/split/index" to {"question", "starter_code"}."""
    if row["pass_rate"] != "1.0" or (row["dataset"], row["split"]) in BENCHMARK_SPLITS:
        return None
    problem = questions.get(f"{row['dataset']}/{row['split']}/{row['index']}")
    parts = split_think(row["r1_generation"])
    if not problem or not problem["question"] or parts is None or not row["solution"].strip():
        return None
    conversation_ = conversation(problem["question"], problem["starter_code"], row["solution"])
    conversation_["messages"][1]["reasoning_content"] = parts[0]
    return conversation_


def codeforces_statement(row):
    """A Codeforces row's statement in the layout of LiveCodeBench's stdin problems."""
    parts = [row["description"].strip()]
    for title, key in (("Input", "input_format"), ("Output", "output_format")):
        if row.get(key):
            parts.append(f"{title}\n\n{row[key].strip()}")
    for i, example in enumerate(row.get("examples") or [], 1):
        parts.append(f"Sample Input {i}\n\n{example['input'].strip()}")
        parts.append(f"Sample Output {i}\n\n{example['output'].strip()}")
    if row.get("note"):
        parts.append(f"Note\n\n{row['note'].strip()}")
    return "\n\n".join(parts)


def codeforces_cots(row):
    """A thinking-mode conversation from a codeforces-cots row, or None unless its last
    Python block passes the row's tests."""
    if row.get("finish_reason") != "stop" or row.get("problem_type") != "diff":
        return None
    if row.get("interaction_format") or not row.get("description"):
        return None
    parts = split_think(row["generation"] or "")
    blocks = parts and livecodebench.CODE_BLOCK.findall(parts[1])
    tests = [
        {"input": i, "output": o}
        for kind in ("public_tests", "private_tests")
        for i, o in zip(
            (row.get(kind) or {}).get("input") or [],
            (row.get(kind) or {}).get("output") or [],
            strict=True,
        )
    ][:MAX_TESTS]
    if not blocks or not tests:
        return None
    program = blocks[-1].strip()
    if not _passes({"starter_code": "", "tests": tests}, program):
        return None
    conversation_ = conversation(codeforces_statement(row), "", program)
    conversation_["messages"][1]["reasoning_content"] = parts[0]
    return conversation_


CONVERTERS = {
    "codecontests": codecontests,
    "codeforces_cots": codeforces_cots,
    "leetcode": leetcode,
    "opencodereasoning2": opencodereasoning2,
}
