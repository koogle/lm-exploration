"""Deterministic, original multi-language repairs with executable oracle cases."""

import json
from pathlib import Path

LANGUAGES = ("python", "javascript", "typescript", "go", "rust", "c", "cpp", "java", "bash")
VERSION = "repair-arithmetic-v1"


def expressions(family, k, language):
    def cond(test, yes, no):
        if language == "python":
            return f"({yes} if {test} else {no})"
        if language == "rust":
            return f"(if {test} {{ {yes} }} else {{ {no} }})"
        return f"({test} ? {yes} : {no})"

    definitions = {
        "sum": (f"a + b + {k}", f"a - b + {k}", f"Return a plus b plus {k}."),
        "difference": (f"a - b + {k}", f"a + b + {k}", f"Return a minus b plus {k}."),
        "product": (f"a * b + {k}", f"a + b + {k}", f"Return a times b plus {k}."),
        "weighted": (f"a * {k} + b", f"a + b * {k}", f"Return {k} times a, plus b."),
        "squares": (
            f"a * a + b * b + {k}",
            f"a * a - b * b + {k}",
            f"Return the sum of the squares of a and b, plus {k}.",
        ),
        "double_sum": (
            f"2 * (a + b) + {k}",
            f"a + b + {k}",
            f"Return twice the sum of a and b, plus {k}.",
        ),
        "clamp": (
            cond(f"a < {k}", str(k), "a"),
            cond(f"a > {k}", str(k), "a"),
            f"Return a clamped to a lower bound of {k}. Ignore b.",
        ),
        "threshold": (
            f"{cond(f'a > {k}', '1', '0')} + {cond(f'b > {k}', '1', '0')}",
            f"{cond(f'a >= {k}', '1', '0')} + {cond(f'b >= {k}', '1', '0')}",
            f"Count how many of a and b are strictly greater than {k}.",
        ),
        "absolute_difference": (
            f"{cond('a > b', 'a - b', 'b - a')} + {k}",
            f"a - b + {k}",
            f"Return the absolute difference of a and b, plus {k}.",
        ),
        "maximum": (
            f"{cond('a > b', 'a', 'b')} + {k}",
            f"{cond('a < b', 'a', 'b')} + {k}",
            f"Return the larger of a and b, plus {k}.",
        ),
    }
    return definitions[family]


def oracle(family, a, b, k):
    return {
        "sum": a + b + k,
        "difference": a - b + k,
        "product": a * b + k,
        "weighted": a * k + b,
        "squares": a**2 + b**2 + k,
        "double_sum": 2 * (a + b) + k,
        "clamp": max(a, k),
        "threshold": int(a > k) + int(b > k),
        "absolute_difference": abs(a - b) + k,
        "maximum": max(a, b) + k,
    }[family]


def function(language, expression):
    if language == "python":
        return f"def solve(a, b):\n    return {expression}\n"
    if language == "javascript":
        return f"function solve(a, b) {{\n    return {expression};\n}}\n"
    if language == "typescript":
        return f"function solve(a: number, b: number): number {{\n    return {expression};\n}}\n"
    if language == "go":
        # Go has no conditional expression; the tiny curriculum uses a helper.
        import re

        pattern = r"\(([^()?]+) \? ([^():]+) : ([^()]+)\)"
        expression = re.sub(pattern, r"choose(\1, \2, \3)", expression)
        helper = "func choose(c bool, x, y int) int { if c { return x }; return y }\n"
        return (
            helper if "choose(" in expression else ""
        ) + f"func solve(a, b int) int {{\n    return {expression}\n}}\n"
    if language == "rust":
        return f"fn solve(a: i32, b: i32) -> i32 {{\n    {expression}\n}}\n"
    if language in ("c", "cpp"):
        return f"int solve(int a, int b) {{\n    return {expression};\n}}\n"
    if language == "java":
        return f"static int solve(int a, int b) {{\n    return {expression};\n}}\n"
    if language == "bash":
        return (
            f'solve() {{\n    local a="$1" b="$2"\n    printf \'%s\\n\' "$(({expression}))"\n}}\n'
        )
    raise ValueError(language)


def rows(language):
    families = (
        "sum",
        "difference",
        "product",
        "weighted",
        "squares",
        "double_sum",
        "clamp",
        "threshold",
        "absolute_difference",
        "maximum",
    )
    output = []
    for family in families:
        split = "validation" if family in ("absolute_difference", "maximum") else "train"
        for k in range(2, 10) if split == "train" else (11, 13):
            good, bad, instruction = expressions(family, k, language)
            pairs = [(-7, 2), (0, 0), (3, -4), (k, k), (k + 1, k - 1), (-2, -9), (12, 5)]
            output.append(
                {
                    "id": f"{VERSION}-{language}-{family}-{k}",
                    "group": f"{VERSION}-{family}",
                    "split": split,
                    "language": language,
                    "instruction": instruction
                    + " Preserve the solve function signature. Inputs are small signed integers.",
                    "broken": function(language, bad),
                    "fixed": function(language, good),
                    "source": f"original procedural curriculum {VERSION}, family={family}, parameter={k}",
                    "license": "CC0-1.0",
                    "family": family,
                    "cases": [[a, b, oracle(family, a, b, k)] for a, b in pairs],
                }
            )
    return output


def harness(language, code, cases):
    if language == "python":
        return code + "\n" + "\n".join(f"assert solve({a}, {b}) == {n}" for a, b, n in cases)
    if language in ("javascript", "typescript"):
        return (
            code
            + "\n"
            + "\n".join(
                f"if (solve({a}, {b}) !== {n}) throw new Error('wrong');" for a, b, n in cases
            )
        )
    if language == "bash":
        return (
            "set -eu\n"
            + code
            + "\n".join(f'[ "$(solve {a} {b})" = "{n}" ] || exit 1' for a, b, n in cases)
        )
    if language in ("c", "cpp"):
        return (
            code
            + "\nint main(void) {\n"
            + "\n".join(f"if (solve({a}, {b}) != {n}) return 1;" for a, b, n in cases)
            + "\nreturn 0; }\n"
        )
    if language == "rust":
        return (
            code
            + "\nfn main() {\n"
            + "\n".join(f"assert_eq!(solve({a}, {b}), {n});" for a, b, n in cases)
            + "\n}\n"
        )
    if language == "go":
        return (
            "package main\n"
            + code
            + "\nfunc main() {\n"
            + "\n".join(f'if solve({a}, {b}) != {n} {{ panic("wrong") }}' for a, b, n in cases)
            + "\n}\n"
        )
    if language == "java":
        return (
            "public class Main {\n"
            + code
            + "\npublic static void main(String[] args) {\n"
            + "\n".join(
                f'if (solve({a}, {b}) != {n}) throw new RuntimeException("wrong");'
                for a, b, n in cases
            )
            + "\n}}\n"
        )
    raise ValueError(language)


def commands(language):
    return {
        "python": ("main.py", None, ["python", "main.py"]),
        "javascript": ("main.js", None, ["node", "main.js"]),
        "typescript": (
            "main.ts",
            ["tsc", "--strict", "--target", "es2020", "main.ts"],
            ["node", "main.js"],
        ),
        "go": ("main.go", ["go", "build", "-o", "program", "main.go"], ["./program"]),
        "rust": ("main.rs", ["rustc", "-o", "program", "main.rs"], ["./program"]),
        "c": ("main.c", ["gcc", "-std=c11", "-o", "program", "main.c"], ["./program"]),
        "cpp": ("main.cpp", ["g++", "-std=c++17", "-o", "program", "main.cpp"], ["./program"]),
        "java": ("Main.java", ["javac", "Main.java"], ["java", "Main"]),
        "bash": ("main.sh", ["bash", "-n", "main.sh"], ["bash", "main.sh"]),
    }[language]


def execute(language, code, cases, folder):
    """Run only inside the verification worker/sandbox, never on the training GPU."""
    import subprocess

    filename, build, run = commands(language)
    Path(folder, filename).write_text(harness(language, code, cases))
    for command in ([build] if build else []) + [run]:
        result = subprocess.run(
            command, check=False, cwd=folder, capture_output=True, text=True, timeout=30
        )
        if result.returncode:
            return {
                "passed": False,
                "stage": "build" if command == build else "test",
                "diagnostic": result.stderr[-2000:],
            }
    return {"passed": True, "stage": "test"}


def write_dataset(path, records):
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
