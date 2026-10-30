"""Teacher-written code solutions (rejection sampling) for `code-sft-v1` problems.

A teacher model answers each problem `samples` times; every answer's code is reduced to
its imports, functions and classes (`code_sft.definitions`) and run against the problem's
unit tests in the code sandbox. The shortest passing answer replaces the row's solution,
so the student imitates teacher code that is known to pass. Problems without a passing
answer are dropped. The problem statement, declaration and tests are unchanged.
"""

import re

from training_data.code_sft import declaration, definitions

TEACHER = ("Qwen/Qwen3.5-35B-A3B-FP8", "9d1823d2dee688a6b25e77009dc727688c44936e")
SYSTEM = "You are an expert Python programmer who writes short, correct code."


def messages(row):
    """Chat messages asking the teacher for one solution."""
    kind = "class" if row["declaration"].startswith("class ") else "function"
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"{row['question'].strip()}\n\nWrite the Python {kind} "
            f"`{row['declaration']}` that solves this. Reply with one ```python code block "
            f"holding the complete solution: the imports it needs and the {kind}, with a "
            "one-line docstring. No example usage, tests or explanation.",
        },
    ]


def extract(text, row):
    """The answer's code if it defines the row's declared name, else None."""
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    if len(blocks) != 1:
        return None
    name = re.match(r"(?:def|class) (\w+)", row["declaration"])[1]
    try:
        code = definitions(blocks[0]).strip()
        return code if code and declaration(code, name) else None
    except SyntaxError:
        return None


def choose(row, answers, results, teacher=TEACHER):
    """The row with its shortest passing answer, or None. `results` are sandbox results
    of the non-None `answers`, in order."""
    passing = []
    outcomes = iter(results)
    for answer in answers:
        if answer is not None and next(outcomes)["passed"]:
            passing.append(answer)
    if not passing:
        return None
    return {
        **row,
        "solution": min(passing, key=len),
        "source": f"{row['source']}; solution by {teacher[0]}@{teacher[1]}",
    }
