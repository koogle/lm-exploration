"""ARC-Easy and ARC-Challenge: science questions with lettered answer choices."""

from evaluation.scoring import choice_accuracy


def item(row):
    choices = row["choices"]
    gold = list(choices["label"]).index(row["answerKey"])
    return f"Question: {row['question']}\nAnswer:", [" " + t for t in choices["text"]], gold


def score(model, rows, exemplars, settings, shots):
    return choice_accuracy(model, [item(row) for row in rows]), None
