"""MBPP+ (EvalPlus): MBPP problems with many more test inputs; greedy pass@1."""

import re

from training_data.code_sft import declaration

from evaluation.scoring import code_answers

SANDBOX = True


def problem(row):
    names = re.findall(r"(\w+)\s*\(", row["test_list"][0].removeprefix("assert"))
    signature = next(s for n in names if (s := declaration(row["code"], n)))
    imports = "".join(f"{line}\n" for line in row["test_imports"])
    return {
        "question": row["prompt"],
        "declaration": signature,
        "preamble": imports,
        "tests": f"from solution import *\n{imports}{row['test']}",
    }


def score(model, rows, exemplars, settings, shots, sandbox):
    problems = [problem(row) for row in rows[: settings.get("examples")]]
    return code_answers(model, problems, sandbox, settings["max_new_tokens"])
