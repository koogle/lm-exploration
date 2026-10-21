"""Repetition metrics shared by sampling reports and corpus audits."""

import re


def repetition(items, n):
    """1 - unique n-grams / all n-grams; 0.0 when the sequence has no complete n-gram."""
    grams = [tuple(items[i : i + n]) for i in range(max(0, len(items) - n + 1))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def normalize(text):
    """Collapse whitespace runs to one space, strip and lowercase."""
    return re.sub(r"\s+", " ", text).strip().lower()


def prefix_repetition(items, width, n):
    """Repetition of the first `width` items, or None when the sequence is shorter."""
    return repetition(items[:width], n) if len(items) >= width else None


def sample_metrics(completion, generation_ids):
    """Native-token, word and normalized-character repetition of one completion."""
    text = normalize(completion)
    words = re.findall(r"\b\w+\b", text)
    return {
        "token_rep4": repetition(generation_ids, 4),
        "word_rep4": repetition(words, 4),
        "char_rep16": repetition(text, 16),
        "char_rep16_first128": prefix_repetition(text, 128, 16),
        "char_rep16_first256": prefix_repetition(text, 256, 16),
        "word_rep4_first32": prefix_repetition(words, 32, 4),
    }
