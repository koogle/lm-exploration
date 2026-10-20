"""Equivalent 13-word benchmark matching with an optional C automaton."""

import re


class LexicalScreen:
    def __init__(self, fingerprints):
        self.reference = fingerprints
        self.automaton = None
        try:
            import ahocorasick
        except ImportError:
            return
        if not fingerprints:
            return
        machine = ahocorasick.Automaton()
        for words in fingerprints:
            machine.add_word(" " + " ".join(words) + " ", True)
        machine.make_automaton()
        self.automaton = machine

    def matches(self, text):
        words = re.findall(r"\w+", text.lower())
        if self.automaton is not None:
            return next(self.automaton.iter(" " + " ".join(words) + " "), None) is not None
        return any(
            tuple(words[i : i + 13]) in self.reference for i in range(max(0, len(words) - 12))
        )
