"""Winogrande: which option fills the blank, by partial scoring.

Each option fills the blank, and the shared text after the blank is scored under both
filled contexts (the harness method), so the options' own likelihoods do not compete.
"""

from evaluation.scoring import loglikelihoods


def score(model, rows, exemplars, settings, shots):
    requests = []
    for row in rows:
        sentence = row["sentence"]
        blank = sentence.index("_")
        continuation = model.encode(" " + sentence[blank + 1 :].strip())
        for option in (row["option1"], row["option2"]):
            requests.append((model.encode(sentence[:blank] + option), continuation))
    totals = [total for total, _ in loglikelihoods(model, requests)]
    correct = sum(
        (totals[2 * i] < totals[2 * i + 1]) == (row["answer"] == "2") for i, row in enumerate(rows)
    )
    return {"examples": len(rows), "acc": correct / len(rows)}, None
