"""Verifiable rewards for RL: math answers and code that must pass unit tests.

Math: the completion's last `\\boxed{...}` is compared with the reference answer
by `math_verify` (symbolic and numeric equivalence); 1 if equal, else 0.

Code: the completion's last fenced Python block becomes `solution.py`; the
problem's pytest file runs against it in a fresh directory, in a separate
`python -I` process with a wall-clock timeout and address-space and CPU limits;
1 if every test passes, else 0. Run this only inside a disposable container: the
limits contain runaway programs, not hostile ones.
"""

import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
TIMEOUT_SECONDS = 10
MEMORY_BYTES = 2 << 30


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


def math_reward(completion, answer):
    from math_verify import parse, verify

    boxed = last_boxed(completion)
    if boxed is None:
        return 0.0
    try:
        return float(verify(parse(f"${answer}$"), parse(f"${boxed}$")))
    except Exception:  # noqa: BLE001 - an unparsable answer earns nothing
        return 0.0


def extract_code(completion):
    blocks = CODE_BLOCK.findall(completion)
    return blocks[-1] if blocks else None


# Applies the limits, then becomes pytest: limits are set in the child itself, which
# is safe from a multithreaded parent (unlike subprocess's preexec_fn).
LIMITED_PYTEST = (
    "import resource, sys; "
    f"resource.setrlimit(resource.RLIMIT_AS, ({MEMORY_BYTES}, {MEMORY_BYTES})); "
    f"resource.setrlimit(resource.RLIMIT_CPU, ({TIMEOUT_SECONDS}, {TIMEOUT_SECONDS})); "
    "import pytest; sys.exit(pytest.main(['-x', '-q', '-p', 'no:cacheprovider']))"
)


def code_reward(completion, tests):
    code = extract_code(completion)
    if code is None:
        return 0.0
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, "solution.py").write_text(code)
        Path(folder, "test_solution.py").write_text(tests)
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", LIMITED_PYTEST],
                cwd=folder,
                capture_output=True,
                check=False,
                timeout=TIMEOUT_SECONDS,
                start_new_session=True,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired:
            return 0.0
    return float(result.returncode == 0)


def score(items, workers=32):
    """Rewards for [(kind, completion, reference)] with kind "math" or "code".

    Math runs in worker processes (math_verify's timeouts need a main thread);
    code runs from threads, each test in its own subprocess.
    """
    from concurrent.futures import ProcessPoolExecutor

    rewards = [0.0] * len(items)
    math = [i for i, item in enumerate(items) if item[0] == "math"]
    code = [i for i, item in enumerate(items) if item[0] == "code"]
    if math:
        with ProcessPoolExecutor(min(workers, len(math))) as pool:
            scores = pool.map(
                math_reward, [items[i][1] for i in math], [items[i][2] for i in math], chunksize=16
            )
            for i, value in zip(math, scores, strict=True):
                rewards[i] = value
    if code:
        with ThreadPoolExecutor(workers) as pool:
            scores = pool.map(code_reward, [items[i][1] for i in code], [items[i][2] for i in code])
            for i, value in zip(code, scores, strict=True):
                rewards[i] = value
    return rewards
