"""HumanEval+ (EvalPlus): HumanEval with many more test inputs; greedy pass@1.

The question is the function's docstring and the declaration its signature; the prompt's
code before the function (imports, helpers) is placed before the model's program.
"""

import ast

from training_data.code_sft import declaration

from evaluation.scoring import code_answers

SANDBOX = True


def problem(row):
    name = row["entry_point"]
    tree = ast.parse(row["prompt"])
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    preamble = "".join(row["prompt"].splitlines(keepends=True)[: function.lineno - 1])
    return {
        "question": ast.get_docstring(function) or "",
        "declaration": declaration(row["prompt"], name),
        "preamble": preamble,
        "tests": f"from solution import *\n{row['test']}\ncheck({name})\n",
    }


def score(model, rows, exemplars, settings, shots, sandbox):
    problems = [problem(row) for row in rows[: settings.get("examples")]]
    return code_answers(model, problems, sandbox, settings["max_new_tokens"])
