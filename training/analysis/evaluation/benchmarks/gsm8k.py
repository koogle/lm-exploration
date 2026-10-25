"""GSM8K: grade-school word problems; worked solutions end in `#### N`."""

from evaluation.scoring import few_shot_answers


def problem(row):
    """(question, gold answer text): the solution's final `#### N` line."""
    return row["question"], row["answer"].rsplit("####", 1)[1].strip()


def exemplar(row):
    """(question, worked solution) from the training split."""
    return row["question"], row["answer"]


def score(model, rows, exemplars, settings, shots):
    problems = [problem(row) for row in rows[: settings.get("examples")]]
    shown = [exemplar(row) for row in exemplars[:shots]]
    return few_shot_answers(model, problems, shown, settings["max_new_tokens"])
