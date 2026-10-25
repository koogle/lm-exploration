"""OpenBookQA: the question stem, continued by each answer choice."""

from evaluation.scoring import choice_accuracy


def item(row):
    choices = row["choices"]
    gold = list(choices["label"]).index(row["answerKey"])
    return row["question_stem"], [" " + t for t in choices["text"]], gold


def score(model, rows, exemplars, settings, shots):
    return choice_accuracy(model, [item(row) for row in rows]), None
