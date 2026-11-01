"""LiveCodeBench code generation: prompts, greedy generation and execution grading.

Problems come from `livecodebench/code_generation_lite` (a release file such as
`test6.jsonl`, pinned by revision). Prompts are family-neutral text: `base_prompt`
(LiveCodeBench's plain prompt after one worked example) for pretrained checkpoints, and
`instruction` (its system text and zero-shot question) for post-trained ones, which
each family renders as its own user turn (`trainer/families.py`). The first fenced
Python block of the greedy completion is graded on every public and private test. Stdin problems (AtCoder) run the program
with the test input and compare whitespace-separated output tokens; functional
problems (LeetCode) call the starter code's method on the JSON-decoded argument
lines and compare JSON results. Each test runs in a separate `python -I`
subprocess with a wall-clock timeout and address-space and CPU limits; a problem
passes when all of its tests pass. Run this only inside a disposable container.

Private tests are stored base64(zlib(pickle(json string))); they are unpickled
with an unpickler that refuses every class, so only plain strings load.
"""

import base64
import io
import json
import pickle
import re
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = (
    "You will be given a question (problem specification) and will generate a correct "
    "Python program that matches the specification and passes all tests."
)
STDIN_FORMAT = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not "
    "directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and "
    "writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```"
)
STARTER_FORMAT = (
    "You will use the following starter code to write the solution to the problem and "
    "enclose your code within delimiters.\n```python\n{starter}\n```"
)
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
TIMEOUT_SECONDS = 6
MEMORY_BYTES = 2 << 30


class _StringsOnly(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def private_tests(encoded):
    raw = zlib.decompress(base64.b64decode(encoded))
    return json.loads(_StringsOnly(io.BytesIO(raw)).load())


def load(path, since=None):
    """Problems with their tests; `since` (ISO date) keeps problems from that date on."""
    problems = []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if since and row["contest_date"][:10] < since:
                continue
            row["tests"] = json.loads(row["public_test_cases"]) + private_tests(
                row["private_test_cases"]
            )
            del row["public_test_cases"], row["private_test_cases"]
            problems.append(row)
    return problems


def question(problem):
    form = (
        STARTER_FORMAT.format(starter=problem["starter_code"])
        if problem["starter_code"]
        else STDIN_FORMAT
    )
    return f"Question: {problem['question_content']}\n\n{form}"


# One worked example per test mode for base models, which otherwise often echo the
# placeholder block; LiveCodeBench's base-model prompts likewise carry one example.
STDIN_EXAMPLE = {
    "question_content": (
        "You are given N integers A_1, ..., A_N. Print the difference between the largest "
        "and the smallest of them.\n\nInput\n\nN\nA_1 A_2 ... A_N\n\nOutput\n\n"
        "Print the answer.\n\nSample Input 1\n\n3\n4 9 2\n\nSample Output 1\n\n7"
    ),
    "starter_code": "",
    "answer": (
        "n = int(input())\nvalues = list(map(int, input().split()))\n"
        "print(max(values) - min(values))"
    ),
}
STARTER_EXAMPLE = {
    "question_content": (
        "You are given a string s. Return the number of vowels ('a', 'e', 'i', 'o', 'u') "
        'in s.\n\nExample 1:\n\nInput: s = "banana"\nOutput: 3'
    ),
    "starter_code": "class Solution:\n    def countVowels(self, s: str) -> int:\n        ",
    "answer": (
        "class Solution:\n    def countVowels(self, s: str) -> int:\n"
        "        return sum(1 for c in s if c in 'aeiou')"
    ),
}


def _base_turn(problem):
    return f"{question(problem)}\n\nAnswer: (use the provided format with backticks)\n\n"


def base_prompt(problem):
    """LiveCodeBench's plain prompt for base models, after one worked example of the
    same test mode; the answer starts with a code fence."""
    example = STARTER_EXAMPLE if problem["starter_code"] else STDIN_EXAMPLE
    shot = f"{_base_turn(example)}```python\n{example['answer']}\n```\n\n"
    return f"{SYSTEM}\n\n{shot}{_base_turn(problem)}"


def instruction(problem):
    """LiveCodeBench's system text and zero-shot question, as one instruction."""
    return f"{SYSTEM}\n\n{question(problem)}"


def extract_code(completion):
    blocks = CODE_BLOCK.findall(completion)
    if blocks:
        return blocks[0]
    # A completion cut off inside its first block still yields the code so far.
    start = completion.find("```")
    return (
        completion[start:].split("\n", 1)[1] if start >= 0 and "\n" in completion[start:] else None
    )


def finished(text):
    """True once a completion holds a closed code block (generation can stop)."""
    return bool(CODE_BLOCK.search(text))


RUNNER = f"""
import json, resource, sys
resource.setrlimit(resource.RLIMIT_AS, ({MEMORY_BYTES}, {MEMORY_BYTES}))
resource.setrlimit(resource.RLIMIT_CPU, ({TIMEOUT_SECONDS}, {TIMEOUT_SECONDS}))
sys.setrecursionlimit(100000)
mode, method = sys.argv[1], sys.argv[2]
source = open("solution.py").read()
if mode == "stdin":
    exec(compile(source, "solution.py", "exec"), {{"__name__": "__main__"}})
else:
    scope = {{"__name__": "solution"}}
    exec(compile("from typing import *\\nfrom collections import *\\nimport math, heapq, bisect, itertools, functools\\n" + source, "solution.py", "exec"), scope)
    args = [json.loads(line) for line in sys.stdin.read().splitlines() if line.strip()]
    print(json.dumps(getattr(scope["Solution"](), method)(*args)))
"""


def _method(starter):
    names = re.findall(r"def\s+(\w+)\s*\(\s*self", starter)
    return names[-1] if names else ""


def _same(expected, actual, functional):
    if functional:
        try:
            return json.loads(expected) == json.loads(actual)
        except json.JSONDecodeError:
            return expected.strip() == actual.strip()
    return expected.split() == actual.split()


def grade(problem, completion):
    """1.0 when the completion's code passes every test, else 0.0."""
    code = extract_code(completion)
    if code is None:
        return 0.0
    functional = bool(problem["starter_code"])
    method = _method(problem["starter_code"]) if functional else ""
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, "solution.py").write_text(code)
        Path(folder, "runner.py").write_text(RUNNER)
        for test in problem["tests"]:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "runner.py",
                        "functional" if functional else "stdin",
                        method,
                    ],
                    cwd=folder,
                    input=test["input"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=TIMEOUT_SECONDS + 2,
                    start_new_session=True,
                    env={"PATH": ""},
                )
            except subprocess.TimeoutExpired:
                return 0.0
            if result.returncode != 0 or not _same(test["output"], result.stdout, functional):
                return 0.0
    return 1.0


def grade_all(problems, completions, workers=16):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(grade, problems, completions))


def summary(problems, scores):
    out = {"pass@1": sum(scores) / len(scores), "problems": len(scores)}
    for key in ("difficulty", "platform"):
        groups = {}
        for problem, score in zip(problems, scores, strict=True):
            groups.setdefault(problem[key], []).append(score)
        out[key] = {k: sum(v) / len(v) for k, v in sorted(groups.items())}
    return out
