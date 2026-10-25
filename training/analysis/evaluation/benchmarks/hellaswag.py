"""HellaSwag: pick the ending of an activity description (EleutherAI harness format)."""

import re

from evaluation.scoring import choice_accuracy


def clean(text):
    """The harness's cleanup of WikiHow markup such as `[title]` and `[header]`."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def item(row):
    context = clean(row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize())
    return context, [" " + clean(ending) for ending in row["endings"]], int(row["label"])


def score(model, rows, exemplars, settings, shots):
    return choice_accuracy(model, [item(row) for row in rows]), None
