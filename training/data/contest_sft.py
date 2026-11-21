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
"""

import ast
import json

from evaluation.benchmarks import livecodebench

CODECONTESTS = ("deepmind/code_contests", "802411c3010cb00d1b05bad57ca77365a3c699d6")
LEETCODE = ("newfacade/LeetCodeDataset", "215604aeed660029df7de2fea5a4d7b6ed476a08")
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


CONVERTERS = {"codecontests": codecontests, "leetcode": leetcode}
