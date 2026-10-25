"""Freeze a staged raw database for loader qualification; never approve pending audits."""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .growing import COMPLETE_RECORD, GROWING_FORMAT, write_json
from .raw_token_stream import token_targets
from .subword import TOKENIZER_SHA256, digest


def describe(db):
    """The staging identity and per-source manifest entries with current split counts."""
    identity = json.loads(db.execute("SELECT value FROM settings WHERE id=1").fetchone()[0])
    sources = {}
    for entry in identity["recipe"]["sources"]:
        counts = dict(
            db.execute(
                """SELECT split, SUM(loss_tokens) FROM documents
            WHERE source=? AND conflict=0 GROUP BY split""",
                (entry["name"],),
            )
        )
        sources[entry["name"]] = {
            "weight_percent": entry["weight_percent"],
            "repo": entry["repo"],
            "revision": entry["revision"],
            "splits": {
                split: {"loss_tokens": counts.get(split, 0)} for split in ("train", "validation")
            },
        }
        if "max_passes" in entry:
            sources[entry["name"]]["max_passes"] = entry["max_passes"]
    return identity, sources


def describe_database(path):
    """`describe` for the database file at `path`, opened read-only."""
    with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)) as db:
        return describe(db)


def export_growing(folder, destination, pending):
    """Write a growing snapshot over the persisted database that preparation still appends to.

    The manifest is written once; its counts describe the prefix available at launch.
    The database stays at `folder/raw.sqlite` (see `data/growing.py`), and
    `complete_growing` records the final database when preparation is done.
    """
    folder, destination = Path(folder), Path(destination)
    if (destination / "manifest.json").exists():
        raise ValueError("Snapshot already exists")
    identity, sources = describe_database(folder / "raw.sqlite")
    manifest = {
        "format": GROWING_FORMAT,
        "training_ready": False,
        "max_passes": 2,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "staging_identity": identity,
        "sources": sources,
        "pending": pending,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def complete_growing(destination, **details):
    """Record the final database preparation persisted into a growing snapshot.

    Later preparation in the same version cannot change this copy, which every later
    training segment reads (see `data/growing.py`).
    """
    destination = Path(destination)
    _, sources = describe_database(destination / "raw.sqlite")
    database = {"path": "raw.sqlite", "sha256": digest(destination / "raw.sqlite")}
    record = {"database": database, "sources": sources, **details}
    write_json(destination / COMPLETE_RECORD, record)
    return record


def export_raw(folder, destination, *, max_passes=None, target_tokens=None):
    """Freeze the staging database with an immutable manifest.

    `max_passes` optionally caps named sources at one pass (`{"name": 1}`).
    `target_tokens` optionally gives every source an exact training-target budget,
    allocated from the recipe weights and summing to that total.
    """
    folder, destination = Path(folder), Path(destination)
    # Must run after the staging worker has stopped, against the committed volume.
    if destination.exists():
        raise ValueError("Snapshot already exists")
    source = sqlite3.connect(f"{(folder / 'raw.sqlite').resolve().as_uri()}?mode=ro", uri=True)
    try:
        identity, sources = describe(source)
        if set(max_passes or {}) - set(sources):
            raise ValueError("Pass caps name an unknown source")
        for name, cap in (max_passes or {}).items():
            sources[name]["max_passes"] = cap
        if target_tokens is not None:
            weights = {name: entry["weight_percent"] for name, entry in sources.items()}
            for name, budget in token_targets(weights, target_tokens).items():
                sources[name]["target_tokens"] = budget
        destination.mkdir(parents=True)
        target = sqlite3.connect(destination / "raw.sqlite")
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    manifest = {
        "format": "raw-text-v1",
        "training_ready": False,
        "max_passes": 2,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "staging_identity": identity,
        "database": {"path": "raw.sqlite", "sha256": digest(destination / "raw.sqlite")},
        "sources": sources,
        "pending": ["capacity, split reconciliation, deduplication and provenance audits"],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument(
        "--one-pass", action="append", default=[], metavar="SOURCE", help="Read SOURCE once"
    )
    parser.add_argument("--target-tokens", type=int, help="Exact per-source budgets summing to N")
    args = parser.parse_args()
    export_raw(
        args.folder,
        args.destination,
        max_passes=dict.fromkeys(args.one_pass, 1),
        target_tokens=args.target_tokens,
    )
