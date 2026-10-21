"""Bounded Python/math curriculum: deterministic specs and independent oracles."""

import ast
import hashlib
import json
import math
import random
import subprocess
import sys

FAMILIES = (
    "factorial",
    "sum_powers",
    "multiples",
    "gcd",
    "prime",
    "dedup",
    "prefix",
    "count",
)


def task(index):
    family = FAMILIES[index % len(FAMILIES)]
    group = index // len(FAMILIES)
    # Entire transformation variants are held out, not randomly selected answers.
    variant = group % 4
    split = "validation" if variant == 3 else "train"
    rng = random.Random(20261021 + index)
    k = 5 + group // 4
    m = 3
    cases = []
    if family == "factorial":
        shift = variant
        spec = f"First set t = n + {shift}. Return t! modulo {k * 97 + m}, where t! is the product of all integers from 1 through t inclusive. n is a nonnegative integer. Recall 0! = 1. For n=0 the result is {math.factorial(shift) % (k * 97 + m)}. The addition happens BEFORE the factorial."
        for n in range(20):
            cases.append(([n], math.factorial(n + shift) % (k * 97 + m)))
        signature = "solve(n)"
    elif family == "sum_powers":
        power = 1 + variant
        spec = f"Return the sum of i**{power} for integers i from {k} through n inclusive. Return 0 if n < {k}."
        for n in range(k + 10):
            cases.append(([n], sum(i**power for i in range(k, n + 1))))
        signature = "solve(n)"
    elif family == "multiples":
        remainder = variant % k
        spec = f"Count integers i in the inclusive interval [lo, hi] whose remainder modulo {k} is {remainder}. Return 0 if lo > hi."
        for _ in range(40):
            lo, hi = rng.randint(-30, 30), rng.randint(-30, 30)
            cases.append(([lo, hi], sum(i % k == remainder for i in range(lo, hi + 1))))
        signature = "solve(lo, hi)"
    elif family == "gcd":
        spec = f"Return gcd(abs(a + {variant}), abs(b + {k})). Define gcd(0, 0) = 0. Inputs are integers."
        for a, b in [(0, 0), (-variant, -k), (12, 18), (-12, 18)] + [
            (rng.randint(-50, 50), rng.randint(-50, 50)) for _ in range(30)
        ]:
            cases.append(([a, b], math.gcd(a + variant, b + k)))
        signature = "solve(a, b)"
    elif family == "prime":
        spec = f"Return whether n + {variant + 4 * (k - 5)} is prime. A prime is an integer greater than 1 with exactly two positive divisors."
        for n in range(-5, 65):
            x = n + variant + 4 * (k - 5)
            cases.append(
                ([n], x > 1 and all(x % d for d in range(2, math.isqrt(max(0, x)) + 1)))
            )
        signature = "solve(n)"
    else:
        arrays = [
            [],
            [0],
            [1, 1],
            [3, 1, 3, 2, 1],
            [-2, 0, -2],
            [k - 1, k, k + 1, -k - 1, -k, -k + 1],
        ]
        arrays += [
            [rng.randint(-8, 8) for _ in range(rng.randrange(15))] for _ in range(30)
        ]
        signature = "solve(items)"
        if family == "dedup":
            descriptions = [
                "keep first occurrences in their original order",
                "keep last occurrences in their original order",
                "sort unique values in ascending order",
                "sort unique values in descending order",
            ]
            spec = f"Filter out values less than {-k}, then {descriptions[variant]}. Return a new list; do not mutate items."
            for xs in arrays:
                kept = [x for x in xs if x >= -k]
                if variant == 0:
                    out = list(dict.fromkeys(kept))
                elif variant == 1:
                    out = list(reversed(list(dict.fromkeys(reversed(kept)))))
                else:
                    out = sorted(set(kept), reverse=variant == 3)
                cases.append(([xs], out))
        elif family == "prefix":
            spec = f"Return a new list of inclusive prefix sums of (x + {variant}) * {k} for each x in items. Do not mutate items."
            for xs in arrays:
                values = [(x + variant) * k for x in xs]
                cases.append(([xs], [sum(values[: i + 1]) for i in range(len(xs))]))
        else:
            comparisons = [
                "strictly less than",
                "less than or equal to",
                "strictly greater than",
                "greater than or equal to",
            ]
            spec = f"Count values in items that are {comparisons[variant]} {k}. Do not mutate items."
            for xs in arrays:
                predicates = [
                    lambda x: x < k,
                    lambda x: x <= k,
                    lambda x: x > k,
                    lambda x: x >= k,
                ]
                cases.append(([xs], sum(predicates[variant](x) for x in xs)))
    prompt = f"Implement {signature} in Python. {spec}"
    identity = hashlib.sha256(prompt.encode()).hexdigest()
    return {
        "id": identity,
        "index": index,
        "family": family,
        "variant": variant,
        "split": split,
        "prompt": prompt,
        "cases": cases,
    }


SAFE_NAMES = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "reversed",
    "set",
    "sorted",
    "sum",
    "tuple",
    "zip",
    "solve",
}
SAFE_METHODS = {"append", "add", "get", "count", "index", "reverse", "sort", "copy"}


def check_ast(code):
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected exactly one function")
    fn = tree.body[0]
    if (
        fn.name != "solve"
        or fn.decorator_list
        or fn.args.defaults
        or fn.args.kw_defaults
        or fn.returns
    ):
        raise ValueError("Invalid function contract")
    forbidden = (
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.AsyncFunctionDef,
        ast.With,
        ast.AsyncWith,
        ast.Global,
        ast.Nonlocal,
        ast.Lambda,
        ast.Yield,
        ast.YieldFrom,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise TypeError("Unsupported syntax")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("Private identifiers prohibited")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_METHODS:
            raise ValueError("Unsupported attribute")
        if isinstance(node, ast.Call) and not (
            (isinstance(node.func, ast.Name) and node.func.id in SAFE_NAMES)
            or (isinstance(node.func, ast.Attribute) and node.func.attr in SAFE_METHODS)
        ):
            raise ValueError("Unsupported call")
    return tree


# Separate process, isolated Python, restricted builtins and resource limits.
# No teacher code runs in the GPU process. This supports only pure function tasks.
RUNNER = r"""
import builtins, copy, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
payload = json.load(sys.stdin)
names = payload['names']
scope = {'__builtins__': {n: getattr(builtins, n) for n in names if n != 'solve'}}
try:
    exec(compile(payload['code'], '<solution>', 'exec'), scope)
    for args, expected in payload['cases']:
        before = copy.deepcopy(args)
        actual = scope['solve'](*args)
        if type(actual) is not type(expected) or actual != expected or args != before:
            raise ValueError('wrong output or mutated input')
    print(json.dumps({'passed': True, 'tests': len(payload['cases'])}))
except BaseException as e:
    print(json.dumps({'passed': False, 'reason': type(e).__name__}))
"""


def verify(code, cases):
    try:
        if not isinstance(code, str) or len(code) > 12000:
            raise ValueError("Invalid code length")
        check_ast(code)
        result = subprocess.run(
            [sys.executable, "-I", "-c", RUNNER],
            input=json.dumps(
                {"code": code, "cases": cases, "names": sorted(SAFE_NAMES)}
            ),
            text=True,
            check=False,
            capture_output=True,
            timeout=5,
            env={},
            cwd="/tmp",
        )
        if result.returncode:
            return {"passed": False, "reason": "process_limit_or_error"}
        return json.loads(result.stdout)
    except (TypeError, ValueError, SyntaxError, subprocess.TimeoutExpired) as error:
        return {"passed": False, "reason": type(error).__name__}


def training_text(record):
    answer = record["answer"]
    return f"### Problem\n{record['prompt']}\n### Python\n{answer['code']}\n### Explanation\n{answer['explanation']}\n"
