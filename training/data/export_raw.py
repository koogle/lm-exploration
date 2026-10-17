"""Freeze a staged raw database for loader qualification; never approve pending audits."""

import argparse
import json
import sqlite3
from pathlib import Path

from .subword import TOKENIZER_SHA256, digest


def export_raw(folder, destination):
    folder, destination = Path(folder), Path(destination)
    # Must run after the staging worker has stopped, against the committed volume.
    if destination.exists():
        raise ValueError("Snapshot already exists")
    destination.mkdir(parents=True)
    source = sqlite3.connect(f"{(folder / 'raw.sqlite').resolve().as_uri()}?mode=ro", uri=True)
    target = sqlite3.connect(destination / "raw.sqlite")
    try:
        source.backup(target)
        identity = json.loads(source.execute("SELECT value FROM settings WHERE id=1").fetchone()[0])
        sources = {}
        for entry in identity["recipe"]["sources"]:
            counts = dict(
                source.execute(
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
                    split: {"loss_tokens": counts.get(split, 0)}
                    for split in ("train", "validation")
                },
            }
    finally:
        source.close()
        target.close()
    manifest = {
        "format": "scholarly-raw-v1",
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
    args = parser.parse_args()
    export_raw(args.folder, args.destination)
