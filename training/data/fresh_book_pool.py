"""Capacity-aware redistribution within books; never change other category budgets."""

import json
from fractions import Fraction
from pathlib import Path

from .fresh_raw_corpus import FreshRawCorpus
from .scholarly import canonical

BOOK_SOURCES = frozenset(
    ("pressbooks", "libretexts", "doab", "gutenberg", "pre_1929_books", "library_of_congress")
)


def allocate_books(original, exhausted_caps, books=BOOK_SOURCES):
    """Water-fill exact integer budgets, capping exhausted book collections only."""
    books = set(books) & original.keys()
    if not books or set(exhausted_caps) - books:
        raise ValueError("Expected known book sources and book-only caps")
    if any(type(v) is not int or v < 0 for v in exhausted_caps.values()):
        raise ValueError("Book capacity must be nonnegative integer targets")
    budget = sum(original[n] for n in books)
    fixed = {}
    while True:
        free = books - fixed.keys()
        remaining = budget - sum(fixed.values())
        if not free:
            if remaining:
                raise ValueError("All book collections exhausted below the fresh books budget")
            return {**original, **fixed}
        denominator = sum(original[n] for n in free)
        exact = {n: Fraction(remaining * original[n], denominator) for n in free}
        bound = {n for n in free if n in exhausted_caps and exact[n] > exhausted_caps[n]}
        if bound:
            fixed.update({n: exhausted_caps[n] for n in bound})
            continue
        allocated = {n: int(v) for n, v in exact.items()}
        leftover = remaining - sum(allocated.values())
        for name in sorted(free, key=lambda n: (-(exact[n] - allocated[n]), n))[:leftover]:
            allocated[name] += 1
        # Largest-remainder rounding cannot cross an integer cap whose exact
        # allocation was already <= that cap.
        return {**original, **fixed, **allocated}


class FreshBookPoolCorpus(FreshRawCorpus):
    def __init__(self, folder, recipe, weights, targets, **kwargs):
        self.original_targets = dict(targets)
        super().__init__(folder, recipe, weights, targets, **kwargs)
        policy = canonical(
            {
                "policy": "fresh-book-pool-v1",
                "original_targets": targets,
                "reserve_percent": 5,
                "max_passes": 2,
            }
        )
        with self.db:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS allocation_policy (id INTEGER PRIMARY KEY,value TEXT)"
            )
            prior = self.db.execute("SELECT value FROM allocation_policy WHERE id=1").fetchone()
            if prior and prior[0] != policy:
                raise ValueError("Fresh book allocation policy changed")
            self.db.execute("INSERT OR IGNORE INTO allocation_policy VALUES (1,?)", (policy,))

    def report(self, target_tokens):
        caps = {}
        for source in self.recipe["sources"]:
            name = source["name"]
            if name in BOOK_SOURCES and self.cursor(name)[0] >= len(source["files"]):
                row = self.db.execute(
                    "SELECT tokens FROM fresh_counts WHERE source=?", (name,)
                ).fetchone()
                # Reserve is inside the two-pass cap, including odd target rounding.
                caps[name] = 2 * ((row[0] if row else 0) * 100 // 105)
        self.targets = allocate_books(self.original_targets, caps)
        self.weights = {n: Fraction(t * 100, target_tokens) for n, t in self.targets.items()}
        result = super().report(target_tokens)
        result["book_allocation"] = {
            "policy": "fresh-book-pool-v1",
            "exhausted_caps": caps,
            "original_targets": self.original_targets,
            "allocated_targets": self.targets.copy(),
        }
        return result

    def export(self, destination):
        manifest = super().export(destination)
        manifest["book_allocation"] = self.report(sum(self.original_targets.values()))[
            "book_allocation"
        ]
        (Path(destination) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
