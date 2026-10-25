"""SVAMP: one-step arithmetic word problems with numeric answers."""

from evaluation.scoring import few_shot_answers


def question(row):
    return f"{row['Body'].strip()} {row['Question'].strip()}"


def answer(row):
    value = float(row["Answer"])
    return str(int(value)) if value.is_integer() else str(value)


def score(model, rows, exemplars, settings, shots):
    problems = [(question(row), answer(row)) for row in rows[: settings.get("examples")]]
    shown = [(question(row), answer(row)) for row in exemplars[:shots]]
    return few_shot_answers(model, problems, shown, settings["max_new_tokens"])
