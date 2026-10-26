"""LAMBADA: predict a passage's last word."""

import math

from evaluation.scoring import loglikelihoods


def score(model, rows, exemplars, settings, shots):
    """Last-word accuracy (every target token greedy) and last-word perplexity."""
    requests = []
    for row in rows:
        context, word = row["text"].rsplit(" ", 1)
        requests.append((model.encode(context), model.encode(" " + word)))
    scores = loglikelihoods(model, requests)
    tokens = sum(len(continuation) for _, continuation in requests)
    return {
        "examples": len(rows),
        "acc": sum(greedy for _, greedy in scores) / len(rows),
        "perplexity": math.exp(-sum(total for total, _ in scores) / tokens),
    }, None
