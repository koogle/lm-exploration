"""Lexical identity helpers for corpus deduplication and benchmark screening."""

import hashlib
import re


def normalized_hash(text):
    """SHA-256 of the text with all whitespace removed (catches reformatted copies)."""
    return hashlib.sha256("".join(text.split()).encode()).hexdigest()


def shingles(text, width=13):
    """Lower-cased word n-grams; 13 words is the usual benchmark-contamination window."""
    words = re.findall(r"\w+", text.lower())
    return (tuple(words[i : i + width]) for i in range(max(0, len(words) - width + 1)))


class LexicalScreen:
    """Flags text that shares any 13-word shingle with a reference fingerprint set."""

    def __init__(self, fingerprints):
        self.fingerprints = set(fingerprints)

    def matches(self, text):
        return any(shingle in self.fingerprints for shingle in shingles(text))
