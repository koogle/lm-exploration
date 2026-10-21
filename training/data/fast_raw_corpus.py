"""Parallel bounded fetching/tokenization with a single transactional SQL writer.

The original RawCorpus data identity, filters, tables and repetition policy are
preserved. Source batches commit in deterministic round-robin order. Workers
never write SQLite. Uncommitted prefetched rows can be replayed after a restart.
"""

import hashlib
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

from .persistent_stream import PersistentStream
from .raw_corpus import RawCorpus
from .simple_llm import adapt, canonical, stream_rows
from .subword import FIRST_CONTENT_TOKEN_ID


class FastRawCorpus(RawCorpus):
    def __init__(self, folder, recipe, *, workers=3, batch_size=64, reader=stream_rows):
        super().__init__(folder, recipe)
        if workers < 1 or batch_size < 1:
            raise ValueError("Positive worker and batch sizes required")
        self.batch_size = batch_size
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.streams = {s["name"]: PersistentStream(reader) for s in recipe["sources"]}
        self.futures = {}
        self.scanned = 0
        # One migration scan; subsequent progress checks use trigger-maintained
        # counts rather than repeatedly scanning millions of stored documents.
        with self.db:
            self.db.execute("BEGIN")
            if not self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fast_counts'"
            ).fetchone():
                self.db.execute("""CREATE TABLE fast_counts (
                    source TEXT, split TEXT, tokens INTEGER, documents INTEGER,
                    PRIMARY KEY(source,split))""")
                self.db.execute("""INSERT INTO fast_counts
                    SELECT source,split,SUM(loss_tokens),COUNT(*) FROM documents
                    WHERE conflict=0 GROUP BY source,split""")
            self.db.executescript("""
                CREATE TRIGGER IF NOT EXISTS fast_insert AFTER INSERT ON documents
                WHEN NEW.conflict=0 BEGIN
                    INSERT INTO fast_counts VALUES(NEW.source,NEW.split,NEW.loss_tokens,1)
                    ON CONFLICT(source,split) DO UPDATE SET
                      tokens=tokens+NEW.loss_tokens, documents=documents+1;
                END;
                CREATE TRIGGER IF NOT EXISTS fast_conflict AFTER UPDATE OF conflict ON documents
                WHEN OLD.conflict=0 AND NEW.conflict=1 BEGIN
                    UPDATE fast_counts SET tokens=tokens-OLD.loss_tokens,documents=documents-1
                    WHERE source=OLD.source AND split=OLD.split;
                END;
            """)

    def counts(self, name):
        return dict(
            self.db.execute(
                "SELECT split,tokens FROM fast_counts WHERE source=? AND documents>0", (name,)
            )
        )

    def cursor(self, name):
        return self.db.execute(
            "SELECT shard,row FROM cursors WHERE source=?", (name,)
        ).fetchone() or (0, 0)

    def _fetch(self, source, cursor):
        for attempt in range(4):
            try:
                return self._fetch_once(source, cursor)
            except Exception as exc:
                network_error = isinstance(exc, (OSError, EOFError)) or type(
                    exc
                ).__module__.startswith(
                    ("requests", "urllib3", "httpx", "httpcore", "huggingface_hub")
                )
                if not network_error or attempt == 3:
                    raise
                self.streams[source["name"]].close()
                time.sleep(min(2**attempt, 8))

    def _fetch_once(self, source, cursor):
        items = []
        total_bytes = 0
        shard_index, start = cursor
        stream = self.streams[source["name"]]
        while shard_index < len(source["files"]):
            rows = stream(source, source["files"][shard_index], start)
            try:
                for row_index, row in rows:
                    document, reason = adapt(source, row)
                    item = {
                        "row": row_index,
                        "shard": shard_index,
                        "document": document,
                        "reason": reason,
                    }
                    if document is not None:
                        content = document["text"].encode()
                        item["hash"] = hashlib.sha256(content).hexdigest()
                        total_bytes += len(content)
                    items.append(item)
                    start = row_index + 1
                    if len(items) >= self.batch_size or total_bytes >= 4 * 1024**2:
                        break
                else:
                    shard_index += 1
                    start = 0
                    continue
                break
            finally:
                rows.close()
        pending = [
            item
            for item in items
            if item["document"] is not None and item["document"]["split"] != "test"
        ]
        encodings = (
            self.tokenizer.encode_batch(
                [item["document"]["text"] for item in pending], add_special_tokens=False
            )
            if pending
            else []
        )
        for item, encoding in zip(pending, encodings):
            ids = encoding.ids
            item["count"] = len(ids) + 1 if ids else 0
            if any(i < FIRST_CONTENT_TOKEN_ID for i in ids):
                item["token_reason"] = "special_token_in_content"
            else:
                item["raw"] = zlib.compress(item["document"]["text"].encode()) if ids else b""
        return {"items": items, "cursor": (shard_index, start), "start": cursor}

    def _write(self, source, batch):
        name = source["name"]
        if self.cursor(name) != batch["start"]:
            raise ValueError("Prefetch cursor differs from committed database")
        # Documents, decisions, counts and the durable cursor commit together.
        with self.db:
            for item in batch["items"]:
                document, reason = item["document"], item["reason"]
                if document is not None:
                    key = item["hash"]
                    previous = self.db.execute(
                        "SELECT split FROM documents WHERE hash=?", (key,)
                    ).fetchone()
                    if previous:
                        reason = "exact_duplicate"
                        if previous[0] != document["split"]:
                            self.db.execute("UPDATE documents SET conflict=1 WHERE hash=?", (key,))
                            reason = "cross_split_duplicate"
                    elif item.get("token_reason"):
                        reason = item["token_reason"]
                    else:
                        count = item.get("count", 0)
                        provenance = {
                            "document": document["document"],
                            "upstream": document["provenance"],
                            "repo": source["repo"],
                            "revision": source["revision"],
                            "shard": source["files"][item["shard"]],
                            "row": item["row"],
                        }
                        self.db.execute(
                            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,0)",
                            (
                                key,
                                name,
                                document["split"],
                                document["parent"],
                                canonical(provenance),
                                item.get("raw", b""),
                                count,
                            ),
                        )
                        reason = "accepted" if count else "excluded_test"
                self.db.execute(
                    """INSERT INTO decisions VALUES (?,?,1)
                    ON CONFLICT(source,reason) DO UPDATE SET count=count+1""",
                    (name, reason),
                )
            self.db.execute(
                "INSERT OR REPLACE INTO cursors VALUES (?,?,?)", (name, *batch["cursor"])
            )
        self.scanned += len(batch["items"])

    @staticmethod
    def enough(status):
        return status["validation_loss_tokens"] > 0 and (
            status["train_loss_tokens"] >= status["minimum_unique_train_tokens"] * 1.05
            or status["upstream_exhausted"]
            and status["capacity_sufficient"]
        )

    def prepare_parallel(self, target_tokens, max_seconds=20):
        started = time.monotonic()
        before = self.scanned
        sources = sorted(self.recipe["sources"], key=lambda s: len(s["files"]))
        while True:
            statuses = self.report(target_tokens)["sources"]
            needed = [s for s in sources if not self.enough(statuses[s["name"]])]
            for source in needed:
                name = source["name"]
                if statuses[name]["upstream_exhausted"]:
                    raise ValueError(f"{name} exhausted before its required data share")
                if name not in self.futures:
                    self.futures[name] = self.pool.submit(self._fetch, source, self.cursor(name))
            if not needed:
                return {
                    "ready": True,
                    "scanned": self.scanned - before,
                    "seconds": time.monotonic() - started,
                }
            # Scheduling order, not network timing, decides cross-source duplicate
            # ownership. All needed sources fetch/tokenize concurrently.
            for source in needed:
                name = source["name"]
                batch = self.futures.pop(name).result()
                self._write(source, batch)
                status = self.report(target_tokens)["sources"][name]
                if not self.enough(status) and not status["upstream_exhausted"]:
                    self.futures[name] = self.pool.submit(self._fetch, source, self.cursor(name))
            if time.monotonic() - started >= max_seconds:
                return {
                    "ready": False,
                    "scanned": self.scanned - before,
                    "seconds": time.monotonic() - started,
                }

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        for stream in self.streams.values():
            stream.close()
        super().close()
