"""One global merge with RAM-resident hash indexes and bounded SQLite page caches.

Kept separate from the immutable running partition implementation so changing merge
performance cannot invalidate workers' saved ingestion identities.
"""

import hashlib
import sqlite3
import time
import zlib
from pathlib import Path

from .readiness_audit import normalized_hash
from .simple_llm import canonical


def merge_partitions(central, partitions, commit=lambda: None):
    db = sqlite3.connect(central)
    db.execute("PRAGMA cache_size=-4194304")  # 4GiB within the 32GiB merge container.
    db.executescript("""
      CREATE TABLE IF NOT EXISTS partition_merges (identity TEXT PRIMARY KEY);
      CREATE TABLE IF NOT EXISTS code_norm_rows (normalized TEXT PRIMARY KEY,hash TEXT,split TEXT);
      CREATE TABLE IF NOT EXISTS merge_metadata (id INTEGER PRIMARY KEY,value TEXT);
    """)
    before = db.execute(
        "SELECT COALESCE(SUM(loss_tokens),0) FROM documents WHERE source='code' AND split='train' AND conflict=0"
    ).fetchone()[0]
    if not db.execute("SELECT 1 FROM merge_metadata WHERE id=1").fetchone():
        with db:
            for key, blob, split in db.execute(
                "SELECT hash,text,split FROM documents NOT INDEXED WHERE source='code'"
            ):
                db.execute(
                    "INSERT OR IGNORE INTO code_norm_rows VALUES (?,?,?)",
                    (normalized_hash(zlib.decompress(blob).decode()), key, split),
                )
            db.execute("INSERT INTO merge_metadata VALUES (1,'serial norms seeded')")
    # Hash-only indexes fit comfortably alongside the two SQLite page caches.
    # Avoid millions of random FUSE reads when rejecting cross-worker duplicates.
    known = {row[0] for row in db.execute("SELECT hash FROM normalized")}
    code = {
        norm: (key, split)
        for norm, key, split in db.execute("SELECT normalized,hash,split FROM code_norm_rows")
    }
    inserted = 0
    for partition in partitions:
        source = sqlite3.connect(f"file:{Path(partition)}?mode=ro&immutable=1", uri=True)
        source.execute("PRAGMA cache_size=-4194304")
        identity = hashlib.sha256(
            source.execute("SELECT value FROM settings WHERE id=1").fetchone()[0].encode()
        ).hexdigest()
        cursor = source.execute("SELECT shard,row FROM cursor WHERE id=1").fetchone()
        marker = canonical([identity, cursor])
        if db.execute("SELECT 1 FROM partition_merges WHERE identity=?", (marker,)).fetchone():
            source.close()
            continue
        with db:
            for row in source.execute(
                "SELECT d.*,n.normalized FROM documents d JOIN dedup n ON n.hash=d.hash ORDER BY d.rowid"
            ):
                *record, norm = row
                previous = code.get(norm)
                if previous:
                    if previous[1] != record[2]:
                        db.execute("UPDATE documents SET conflict=1 WHERE hash=?", (previous[0],))
                    continue
                if norm in known:
                    continue
                known.add(norm)
                code[norm] = (record[0], record[2])
                db.execute("INSERT INTO normalized VALUES (?)", (norm,))
                db.execute(
                    "INSERT INTO code_norm_rows VALUES (?,?,?)", (norm, record[0], record[2])
                )
                if db.execute(
                    "INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?,?,?)", record
                ).rowcount:
                    inserted += record[6] if record[2] == "train" else 0
            db.execute("INSERT INTO partition_merges VALUES (?)", (marker,))
        source.close()
        commit()
    with db:
        db.execute("DELETE FROM stats")
        for kind, field in [("language", "language"), ("task", "task_family")]:
            db.execute(
                f"""INSERT INTO stats SELECT ?,json_extract(provenance,'$.{field}'),SUM(loss_tokens),COUNT(*)
              FROM documents WHERE source='code' AND split='train' AND conflict=0 GROUP BY json_extract(provenance,'$.{field}')""",
                (kind,),
            )
    total = db.execute(
        "SELECT COALESCE(SUM(loss_tokens),0) FROM documents WHERE source='code' AND split='train' AND conflict=0"
    ).fetchone()[0]
    db.close()
    commit()
    return {
        "fresh_code_tokens": total,
        "added_unique_train_tokens": total - before,
        "inserted_train_tokens_before_conflict_exclusions": inserted,
        "partitions": len(partitions),
        "updated_unix": time.time(),
    }
