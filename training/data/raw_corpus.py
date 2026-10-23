"""Resumable raw-text staging and exact capacity measurement, without token shards.

Token IDs are ephemeral while counting; training must tokenize selected raw documents.
This measures capacity and exact deduplication, not the outstanding quality audits.
"""

import hashlib
import sqlite3
import time
import zlib
from pathlib import Path

from .simple_llm import SPLIT_VERSION, adapt, canonical, stream_rows
from .subword import FIRST_CONTENT_TOKEN_ID, TOKENIZER_SHA256, digest, load_frozen


class RawCorpus:
    def __init__(self, folder, recipe, tokenizer=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.recipe = recipe
        self.tokenizer = tokenizer or load_frozen()
        self.db = sqlite3.connect(self.folder / "raw.sqlite")
        page_size = self.db.execute("PRAGMA page_size").fetchone()[0]
        self.db.execute(f"PRAGMA max_page_count={64 * 1024**3 // page_size}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS cursors (source TEXT PRIMARY KEY, shard INTEGER, row INTEGER);
            CREATE TABLE IF NOT EXISTS documents (
                hash TEXT PRIMARY KEY, source TEXT, split TEXT, parent TEXT,
                provenance TEXT, text BLOB, loss_tokens INTEGER, conflict INTEGER DEFAULT 0);
            CREATE INDEX IF NOT EXISTS document_split ON documents(source, split, conflict);
            CREATE TABLE IF NOT EXISTS decisions (
                source TEXT, reason TEXT, count INTEGER, PRIMARY KEY(source, reason));
        """)
        identity = canonical(
            {
                "recipe": recipe,
                "tokenizer": TOKENIZER_SHA256,
                "split": SPLIT_VERSION,
                "implementation_sha256": digest(__file__),
                "adapter_sha256": digest(Path(__file__).with_name("simple_llm.py")),
                "tokenizer_loader_sha256": digest(Path(__file__).with_name("subword.py")),
            }
        )
        previous = self.db.execute("SELECT value FROM settings WHERE id=1").fetchone()
        if previous and previous[0] != identity:
            self.db.close()
            raise ValueError("Raw corpus identity changed; use a new directory")
        self.db.execute("INSERT OR IGNORE INTO settings VALUES (1,?)", (identity,))
        self.db.commit()

    def close(self):
        self.db.close()

    def counts(self, name):
        return dict(
            self.db.execute(
                "SELECT split, SUM(loss_tokens) FROM documents WHERE source=? AND conflict=0 GROUP BY split",
                (name,),
            )
        )

    def prepare(self, name, max_records=1000, max_seconds=240, reader=stream_rows):
        source = next(s for s in self.recipe["sources"] if s["name"] == name)
        cursor = self.db.execute(
            "SELECT shard,row FROM cursors WHERE source=?", (name,)
        ).fetchone() or (0, 0)
        scanned = 0
        started = time.monotonic()
        for shard in range(cursor[0], len(source["files"])):
            rows = reader(source, source["files"][shard], cursor[1] if shard == cursor[0] else 0)
            try:
                for row_index, row in rows:
                    document, reason = adapt(source, row, (source["files"][shard], row_index))
                    with self.db:
                        if document is not None:
                            content = document["text"].encode()
                            key = hashlib.sha256(content).hexdigest()
                            previous = self.db.execute(
                                "SELECT split FROM documents WHERE hash=?", (key,)
                            ).fetchone()
                            if previous:
                                reason = "exact_duplicate"
                                if previous[0] != document["split"]:
                                    self.db.execute(
                                        "UPDATE documents SET conflict=1 WHERE hash=?", (key,)
                                    )
                                    reason = "cross_split_duplicate"
                            else:
                                ids = (
                                    []
                                    if document["split"] == "test"
                                    else self.tokenizer.encode(
                                        document["text"], add_special_tokens=False
                                    ).ids
                                )
                                if any(i < FIRST_CONTENT_TOKEN_ID for i in ids):
                                    reason = "special_token_in_content"
                                else:
                                    provenance = {
                                        "document": document["document"],
                                        "upstream": document["provenance"],
                                        "repo": source["repo"],
                                        "revision": source["revision"],
                                        "shard": source["files"][shard],
                                        "row": row_index,
                                    }
                                    self.db.execute(
                                        "INSERT INTO documents VALUES (?,?,?,?,?,?,?,0)",
                                        (
                                            key,
                                            name,
                                            document["split"],
                                            document["parent"],
                                            canonical(provenance),
                                            zlib.compress(content) if ids else b"",
                                            len(ids) + 1 if ids else 0,
                                        ),
                                    )
                                    reason = "accepted" if ids else "excluded_test"
                        self.db.execute(
                            """INSERT INTO decisions VALUES (?,?,1)
                            ON CONFLICT(source,reason) DO UPDATE SET count=count+1""",
                            (name, reason),
                        )
                        self.db.execute(
                            "INSERT OR REPLACE INTO cursors VALUES (?,?,?)",
                            (name, shard, row_index + 1),
                        )
                    scanned += 1
                    if scanned >= max_records or time.monotonic() - started >= max_seconds:
                        return scanned
            finally:
                if hasattr(rows, "close"):
                    rows.close()
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO cursors VALUES (?,?,0)", (name, shard + 1))
        return scanned

    def report(self, target_tokens):
        sources = {}
        for source in self.recipe["sources"]:
            counts = self.counts(source["name"])
            needed = (target_tokens * source["weight_percent"] + 199) // 200
            cursor = self.db.execute(
                "SELECT shard,row FROM cursors WHERE source=?", (source["name"],)
            ).fetchone() or (0, 0)
            sources[source["name"]] = {
                "weight_percent": source["weight_percent"],
                "train_loss_tokens": counts.get("train", 0),
                "validation_loss_tokens": counts.get("validation", 0),
                "minimum_unique_train_tokens": needed,
                "capacity_sufficient": counts.get("train", 0) >= needed,
                "upstream_exhausted": cursor[0] >= len(source["files"]),
                "cursor": cursor,
            }
        capacity = min(
            s["train_loss_tokens"] * 200 // s["weight_percent"] for s in sources.values()
        )
        return {
            "target_tokens": target_tokens,
            "max_passes": 2,
            "supported_mixture_exposures": capacity,
            "training_ready": False,
            "storage": "compressed raw text; no token-ID shards",
            "sources": sources,
            "pending": [
                "tokenizer split reconciliation",
                "global near deduplication",
                "cross-source parent reconciliation",
                "eligibility and contamination review",
                "batch-driven raw-text training loader qualification",
            ],
        }
