"""Resume upstream cursors, but export only new, exact-hash-disjoint training data."""

import json
import math
import sqlite3
from pathlib import Path

from .fast_raw_corpus import FastRawCorpus
from .subword import TOKENIZER_SHA256, digest


def initialize_fresh(folder, parent_manifest, other_databases=()):
    """Atomic clone; old documents are an exclusion ledger, never new capacity."""
    folder, parent_manifest = Path(folder), Path(parent_manifest)
    parent = json.loads(parent_manifest.read_text())
    databases = [parent_manifest.parent / parent["database"]["path"], *map(Path, other_databases)]
    identity = {
        "parent_manifest_sha256": digest(parent_manifest),
        "excluded_databases": [{"sha256": digest(p), "path": str(p)} for p in databases],
        "policy": "all prior document hashes excluded from new training, including heldouts",
    }
    if identity["excluded_databases"][0]["sha256"] != parent["database"]["sha256"]:
        raise ValueError("Original immutable database checksum mismatch")
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / "raw.sqlite"
    if final.exists():
        with sqlite3.connect(final) as db:
            saved = json.loads(db.execute("SELECT value FROM fresh_identity").fetchone()[0])
        if saved != identity:
            raise ValueError("Fresh staging exclusion identity changed")
        return parent["staging_identity"]["recipe"]
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
            target.execute(f"ATTACH DATABASE ? AS old{index}", (str(database),))
            target.execute(
                f"INSERT OR IGNORE INTO excluded_prior SELECT hash FROM old{index}.documents"
            )
        target.commit()
    temporary.replace(final)
    return parent["staging_identity"]["recipe"]


class FreshRawCorpus(FastRawCorpus):
    def __init__(self, folder, recipe, weights, targets, **kwargs):
        self.weights, self.targets = weights, targets
        if set(weights) != {s["name"] for s in recipe["sources"]} or set(weights) != set(targets):
            raise ValueError("Fresh quotas must cover every original source")
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
        if target_tokens != sum(self.targets.values()):
            raise ValueError("Fresh phase target changed")
        sources = {}
        for source in self.recipe["sources"]:
            name = source["name"]
            row = self.db.execute(
                "SELECT tokens FROM fresh_counts WHERE source=?", (name,)
            ).fetchone()
            count = row[0] if row else 0
            minimum = math.ceil(self.targets[name] / 2)
            sources[name] = {
                "weight_percent": float(self.weights[name]),
                "weight_percent_exact": str(self.weights[name]),
                "phase_target_tokens": self.targets[name],
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

    def export(self, destination):
        """Freeze a clean database; no old training text/cursors enter the snapshot."""
        destination = Path(destination)
        status = self.report(sum(self.targets.values()))
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
                overlap = self.db.execute("""SELECT COUNT(*) FROM frozen.documents d
                    JOIN excluded_prior e ON e.hash=d.hash WHERE d.split='train'""").fetchone()[0]
                if overlap:
                    raise ValueError("Prior training/heldout overlap in fresh export")
                sources = {}
                for name, state in status["sources"].items():
                    if not self.targets[name]:
                        continue
                    counts = dict(
                        self.db.execute(
                            """SELECT split,SUM(loss_tokens)
                        FROM frozen.documents WHERE source=? GROUP BY split""",
                            (name,),
                        )
                    )
                    if counts.get("train", 0) != state["train_loss_tokens"]:
                        raise ValueError("Fresh trigger/export counts disagree")
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
                exclusion = json.loads(
                    self.db.execute("SELECT value FROM fresh_identity").fetchone()[0]
                )
                exclusion["excluded_hashes"] = self.db.execute(
                    "SELECT COUNT(*) FROM excluded_prior"
                ).fetchone()[0]
        finally:
            self.db.execute("DETACH DATABASE frozen")
        manifest = {
            "format": "scholarly-raw-v1",
            "training_ready": False,
            "max_passes": 2,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "database": {"path": "raw.sqlite", "sha256": digest(output)},
            "sources": sources,
            "freshness": {**exclusion, "prior_hash_overlap": overlap},
            "staging_identity": json.loads(
                self.db.execute("SELECT value FROM settings WHERE id=1").fetchone()[0]
            ),
            "pending": [
                "global near-duplicate and qualitative audits; authorized experimental-data run"
            ],
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
