"""Prepare a fresh-only raw snapshot: continue upstream reading, export only new text.

For continued pretraining on data the model has not seen. `initialize_fresh` clones the
parent snapshot's staging database, so every source cursor continues where the parent
stopped, and records every prior document hash (the parent's and any further databases)
as an exclusion ledger. Prior documents never count as new capacity. `FreshRawCorpus`
keeps preparing until each source has enough fresh training targets for its share of
the new phase, and `export` freezes a snapshot with only fresh training documents plus
the validation documents (prior and new, so evaluation stays comparable).
"""

import json
import math
import sqlite3
from fractions import Fraction
from pathlib import Path

from .fast_raw_corpus import FastRawCorpus
from .subword import TOKENIZER_SHA256, digest

POLICY = "every prior document hash, including held-out documents, is excluded from training"


def initialize_fresh(folder, parent_manifest, other_databases=()):
    """Clone the parent staging database once; return the parent's recipe.

    Later calls check that the ledger was built from the same inputs.
    """
    folder, parent_manifest = Path(folder), Path(parent_manifest)
    parent = json.loads(parent_manifest.read_text())
    databases = [parent_manifest.parent / parent["database"]["path"], *map(Path, other_databases)]
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / "raw.sqlite"
    manifest_sha = digest(parent_manifest)
    if final.exists():
        # The committed ledger is self-contained; do not reread multi-GB inputs.
        with sqlite3.connect(final) as db:
            saved = json.loads(db.execute("SELECT value FROM fresh_identity").fetchone()[0])
        if saved["parent_manifest_sha256"] != manifest_sha or [
            entry["path"] for entry in saved["excluded_databases"]
        ] != list(map(str, databases)):
            raise ValueError("Fresh staging was built from different inputs")
        return parent["staging_identity"]["recipe"]
    identity = {
        "parent_manifest_sha256": manifest_sha,
        "excluded_databases": [{"sha256": digest(p), "path": str(p)} for p in databases],
        "policy": POLICY,
    }
    if identity["excluded_databases"][0]["sha256"] != parent["database"]["sha256"]:
        raise ValueError("Parent database checksum mismatch")
    temporary = folder / "initializing.sqlite"
    temporary.unlink(missing_ok=True)
    with sqlite3.connect(databases[0]) as source, sqlite3.connect(temporary) as target:
        source.backup(target)
        target.execute("CREATE TABLE excluded_prior (hash TEXT PRIMARY KEY)")
        target.execute("INSERT INTO excluded_prior SELECT hash FROM documents")
        target.execute("CREATE TABLE fresh_identity (value TEXT NOT NULL)")
        target.execute(
            "INSERT INTO fresh_identity VALUES (?)", (json.dumps(identity, sort_keys=True),)
        )
        for index, database in enumerate(databases[1:]):
            target.execute(f"ATTACH DATABASE ? AS prior{index}", (str(database),))
            target.execute(
                f"INSERT OR IGNORE INTO excluded_prior SELECT hash FROM prior{index}.documents"
            )
        target.commit()
    temporary.replace(final)
    return parent["staging_identity"]["recipe"]


class FreshRawCorpus(FastRawCorpus):
    """`recipe` is the parent's (cursors resume); `weights` are the new phase's mix."""

    def __init__(self, folder, recipe, weights, **kwargs):
        self.weights = {name: Fraction(str(share)) for name, share in weights.items()}
        names = {s["name"] for s in recipe["sources"]}
        if set(self.weights) != names or sum(self.weights.values()) != 100:
            raise ValueError("Fresh weights must cover every recipe source and sum to 100")
        super().__init__(folder, recipe, **kwargs)
        with self.db:
            if not self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='excluded_prior'"
            ).fetchone():
                raise ValueError("Initialize the prior exclusion ledger first")
            self.db.execute("""CREATE TABLE IF NOT EXISTS fresh_counts (
                source TEXT PRIMARY KEY, tokens INTEGER, documents INTEGER)""")
            self.db.executescript("""
                CREATE TRIGGER IF NOT EXISTS fresh_insert AFTER INSERT ON documents
                WHEN NEW.conflict=0 AND NEW.split='train'
                  AND NOT EXISTS (SELECT 1 FROM excluded_prior WHERE hash=NEW.hash) BEGIN
                    INSERT INTO fresh_counts VALUES(NEW.source,NEW.loss_tokens,1)
                    ON CONFLICT(source) DO UPDATE SET
                      tokens=tokens+NEW.loss_tokens,documents=documents+1;
                END;
                CREATE TRIGGER IF NOT EXISTS fresh_conflict AFTER UPDATE OF conflict ON documents
                WHEN OLD.conflict=0 AND NEW.conflict=1 AND OLD.split='train'
                  AND NOT EXISTS (SELECT 1 FROM excluded_prior WHERE hash=OLD.hash) BEGIN
                    UPDATE fresh_counts SET tokens=tokens-OLD.loss_tokens,documents=documents-1
                    WHERE source=OLD.source;
                END;
            """)

    def report(self, target_tokens):
        """Fresh capacity per source for `target_tokens` within two passes."""
        sources = {}
        for source in self.recipe["sources"]:
            name = source["name"]
            row = self.db.execute(
                "SELECT tokens FROM fresh_counts WHERE source=?", (name,)
            ).fetchone()
            count = row[0] if row else 0
            minimum = math.ceil(target_tokens * self.weights[name] / 200)
            sources[name] = {
                "weight_percent": float(self.weights[name]),
                "train_loss_tokens": count,
                "validation_loss_tokens": self.counts(name).get("validation", 0),
                "minimum_unique_train_tokens": minimum,
                "capacity_sufficient": count >= minimum,
                "upstream_exhausted": self.cursor(name)[0] >= len(source["files"]),
                "cursor": self.cursor(name),
            }
        return {
            "target_tokens": target_tokens,
            "max_passes": 2,
            "fresh_only": True,
            "sources": sources,
            "ready": all(self.enough(s) for s in sources.values()),
        }

    def export(self, destination, target_tokens):
        """Freeze fresh training documents plus all validation documents."""
        destination = Path(destination)
        status = self.report(target_tokens)
        if not status["ready"]:
            raise ValueError("Fresh capacity incomplete")
        if destination.exists():
            raise ValueError("Refuse snapshot overwrite")
        destination.mkdir(parents=True)
        output = destination / "raw.sqlite"
        self.db.execute("ATTACH DATABASE ? AS frozen", (str(output),))
        try:
            with self.db:
                self.db.execute("""CREATE TABLE frozen.documents (
                    hash TEXT PRIMARY KEY, source TEXT, split TEXT, parent TEXT,
                    provenance TEXT,text BLOB,loss_tokens INTEGER,conflict INTEGER DEFAULT 0)""")
                self.db.execute("""INSERT INTO frozen.documents SELECT * FROM main.documents d
                    WHERE conflict=0 AND (split='validation' OR
                      (split='train' AND NOT EXISTS
                        (SELECT 1 FROM excluded_prior e WHERE e.hash=d.hash)))""")
                self.db.execute(
                    "CREATE INDEX frozen.document_split ON documents(source,split,conflict)"
                )
                sources = {}
                for name, state in status["sources"].items():
                    if not self.weights[name]:
                        continue
                    counts = dict(
                        self.db.execute(
                            "SELECT split,SUM(loss_tokens) FROM frozen.documents "
                            "WHERE source=? GROUP BY split",
                            (name,),
                        )
                    )
                    if counts.get("train", 0) != state["train_loss_tokens"]:
                        raise ValueError("Fresh counts and exported documents disagree")
                    source = next(s for s in self.recipe["sources"] if s["name"] == name)
                    sources[name] = {
                        "weight_percent": float(self.weights[name]),
                        "weight_percent_exact": str(self.weights[name]),
                        "repo": source["repo"],
                        "revision": source["revision"],
                        "splits": {
                            s: {"loss_tokens": counts.get(s, 0)} for s in ("train", "validation")
                        },
                    }
                freshness = json.loads(
                    self.db.execute("SELECT value FROM fresh_identity").fetchone()[0]
                )
                freshness["excluded_hashes"] = self.db.execute(
                    "SELECT COUNT(*) FROM excluded_prior"
                ).fetchone()[0]
        finally:
            self.db.execute("DETACH DATABASE frozen")
        manifest = {
            "format": "raw-text-v1",
            "training_ready": False,
            "max_passes": 2,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "database": {"path": "raw.sqlite", "sha256": digest(output)},
            "sources": sources,
            "freshness": freshness,
            "staging_identity": json.loads(
                self.db.execute("SELECT value FROM settings WHERE id=1").fetchone()[0]
            ),
            "pending": ["global near-duplicate and qualitative audits"],
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
