"""Evaluation sets, prompt formats and completion parsing.

Two task sets, both graded by executing hidden tests:

- ``mbpp``: all 427 sanitized MBPP problems (human-written tests). The tested
  function is renamed to ``solve`` and the problem is phrased in the training
  data's JSON format, with the signature and the first test shown.
- ``curriculum``: exercises from held-out teacher chunks (never trained on). The
  teacher writes asserts from the problem text alone; an item is kept only when
  the teacher's original solution passes all of them. Unkept items are counted:
  the agreement rate estimates how often the training data is self-consistent.
"""

import ast
import json
import re

PREFIX = '{\n  "problem": '
CODE_KEY = ',\n  "code": "'


def student_prompt(problem):
    """Raw completion prompt in the training format, ending inside the code string."""
    return PREFIX + json.dumps(problem) + CODE_KEY


def parse_code_string(completion):
    """Decode the JSON string body that the model writes after ``"code": "``."""
    escaped = False
    for index, char in enumerate(completion):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            try:
                return json.loads('"' + completion[:index] + '"')
            except json.JSONDecodeError:
                return None
    return None


def signature(code, name):
    """The ``def`` line of ``name`` in ``code``, renamed to ``solve``."""
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            line = code.splitlines()[node.lineno - 1].strip()
            return re.sub(rf"^def\s+{re.escape(name)}\b", "def solve", line).rstrip()
    raise ValueError(f"no function {name}")


def tested_function(code, tests):
    defined = {n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}
    called = set()
    for test in tests:
        for node in ast.walk(ast.parse(test)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)
    candidates = sorted(defined & called)
    if len(candidates) != 1:
        raise ValueError(f"ambiguous tested function: {candidates}")
    return candidates[0]


def rename(text, old, new="solve"):
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def mbpp_items(rows):
    """Rows are sanitized MBPP records (task_id, prompt, code, test_imports, test_list)."""
    items = []
    for row in rows:
        tests = list(row["test_list"])
        name = tested_function(row["code"], tests)
        tests = [rename(t, name) for t in tests]
        problem = (
            f"{row['prompt'].strip()} Implement it as `{signature(row['code'], name)}`. "
            f"Example: `{tests[0]}`."
        )
        items.append(
            {
                "id": f"mbpp-{row['task_id']}",
                "set": "mbpp",
                "problem": problem,
                "setup": "\n".join(row["test_imports"]),
                "tests": tests,
                "reference": row["code"] + f"\nsolve = {name}\n",
            }
        )
    return items


def curriculum_problem(record):
    """Held-out teacher problem plus the reference signature; ``None`` if unusable."""
    try:
        sig = signature(record["code"], "solve")
    except (SyntaxError, ValueError):
        return None
    return f"{record['problem'].strip()} Implement it as `{sig}`."


TEST_PROMPT = (
    "Write hidden unit tests for the following Python exercise.\n\n{problem}\n\n"
    'Return only JSON of the form {{"tests": [...]}} with 6 to 8 strings. Each string is '
    "one single-line Python assert statement that calls solve with literal arguments and "
    "compares the result with a literal expected value. Cover normal cases and the edge "
    "cases named in the contract. Work out every expected value carefully; do not use "
    "randomness, I/O or imports."
)

SOLVE_PROMPT = (
    "Solve the following Python exercise.\n\n{problem}\n\n"
    'Return only JSON with one string field "code" containing a complete Python '
    "implementation of solve. Only standard-library imports; no I/O, network or packages."
)


def parse_json_answer(text, field):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1]).get(field)
    except (json.JSONDecodeError, AttributeError):
        return None
    return value


def valid_tests(tests):
    if not isinstance(tests, list):
        return []
    kept = []
    for test in tests:
        if not isinstance(test, str) or "\n" in test.strip():
            continue
        try:
            node = ast.parse(test.strip())
        except SyntaxError:
            continue
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Assert)
            and "solve(" in test
        ):
            kept.append(test.strip())
    return kept


def curriculum_items(records, tests_by_id, reference_results):
    """Keep held-out items whose reference solution passes all teacher tests."""
    items, counts = (
        [],
        {"candidates": 0, "too_few_tests": 0, "reference_failed": 0, "kept": 0},
    )
    for record in records:
        tests = tests_by_id.get(record["id"])
        if tests is None:
            continue
        counts["candidates"] += 1
        if len(tests) < 3:
            counts["too_few_tests"] += 1
            continue
        if not reference_results.get(record["id"]):
            counts["reference_failed"] += 1
            continue
        counts["kept"] += 1
        items.append(
            {
                "id": f"curriculum-{record['id']}",
                "set": "curriculum",
                "problem": curriculum_problem(record),
                "setup": "",
                "tests": tests,
                "reference": record["code"],
            }
        )
    return items, counts
