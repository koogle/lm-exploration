"""Collect multilingual code on parallel CPU workers, then merge with global dedup.

1. `build_exclusions` freezes an immutable ledger from prior corpora: their document
   hashes, the whitespace-normalized hashes of their code, and the split of every
   prior code repository (held-out repositories stay held out).
2. `prepare_partition` runs on each of `workers` CPU workers over a disjoint set of
   pinned shards. Each worker writes its own resumable database.
3. `merge_partitions` merges finished worker databases into one central database
   (one writer) with global normalized dedup. A document whose normalized copy lands
   in a different split marks the earlier copy as a conflict. Replays are idempotent.
"""

import hashlib
import json
import shutil
import sqlite3
import tempfile
import time
import zlib
from collections import Counter
from pathlib import Path

from .lexical_screen import normalized_hash
from .multilingual_code import CODE_REPO, CODE_REVISION, SCHEMA, select, split_for
from .simple_llm import canonical
from .subword import FIRST_CONTENT_TOKEN_ID, TOKENIZER_SHA256, digest, load_frozen

CODE_SOURCES = ("code", "code_python")


def shard_order(files):
    """A fixed pseudo-random order, so early shards are not one alphabetical slice."""
    names = sorted(name for name in files if name.endswith(".parquet"))
    return sorted(names, key=lambda p: hashlib.sha256(f"multilingual-code:{p}".encode()).digest())


def partition_shards(files, start, worker, workers):
    if not 0 <= worker < workers or start < 0:
        raise ValueError("Invalid shard partition")
    return [(i, p) for i, p in enumerate(files) if i >= start and i % workers == worker]


def build_exclusions(prior_databases, folder, files=None):
    """Freeze the exclusion ledger and partition identity for one collection."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder / "identity.json"
    if metadata.exists():
        raise ValueError("Exclusions already exist; use a new collection folder")
    if files is None:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(CODE_REPO, repo_type="dataset", revision=CODE_REVISION)
    temporary = folder / "exclusions.building.sqlite"
    temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    db.executescript("""
        CREATE TABLE hashes (hash TEXT PRIMARY KEY);
        CREATE TABLE normalized (hash TEXT PRIMARY KEY);
        CREATE TABLE repositories (parent TEXT PRIMARY KEY, split TEXT);
    """)
    code = ",".join(f"'{name}'" for name in CODE_SOURCES)
    for index, path in enumerate(prior_databases):
        db.execute(f"ATTACH DATABASE ? AS prior{index}", (str(path),))
        table = f"prior{index}.documents"
        db.execute(f"INSERT OR IGNORE INTO hashes SELECT hash FROM {table}")
        rows = db.execute(
            f"SELECT text FROM {table} WHERE source IN ({code}) AND length(text)>0"
        ).fetchall()
        db.executemany(
            "INSERT OR IGNORE INTO normalized VALUES (?)",
            ((normalized_hash(zlib.decompress(blob).decode()),) for (blob,) in rows),
        )
        db.execute(f"""INSERT INTO repositories
            SELECT lower(parent), CASE WHEN SUM(split!='train')>0 THEN 'heldout' ELSE 'train' END
            FROM {table} WHERE source IN ({code}) GROUP BY lower(parent)
            ON CONFLICT(parent) DO UPDATE SET split=CASE
              WHEN repositories.split!='train' OR excluded.split!='train' THEN 'heldout'
              ELSE 'train' END""")
        db.commit()
        db.execute(f"DETACH DATABASE prior{index}")
    db.close()
    temporary.replace(folder / "exclusions.sqlite")
    identity = {
        "repo": CODE_REPO,
        "revision": CODE_REVISION,
        "tokenizer": TOKENIZER_SHA256,
        "prior_databases": [str(p) for p in prior_databases],
        "files": shard_order(files),
        "exclusions_sha256": digest(folder / "exclusions.sqlite"),
    }
    metadata.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def prepare_partition(folder, exclusions, identity, worker, workers, target, commit=lambda: None):
    """Collect up to `target` new training targets from this worker's shards; resumable."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # Random lookups stay on local disk rather than a network volume.
    local = Path(tempfile.gettempdir()) / f"code-exclusions-{worker}.sqlite"
    if not local.exists() or digest(local) != identity["exclusions_sha256"]:
        shutil.copyfile(exclusions, local)
    if digest(local) != identity["exclusions_sha256"]:
        raise ValueError("Exclusion ledger checksum mismatch")
    old = sqlite3.connect(f"file:{local}?mode=ro&immutable=1", uri=True)
    repositories = dict(old.execute("SELECT parent,split FROM repositories"))
    db = sqlite3.connect(folder / "raw.sqlite")
    db.executescript(
        SCHEMA
        + """
      CREATE TABLE IF NOT EXISTS dedup (normalized TEXT PRIMARY KEY,hash TEXT UNIQUE,split TEXT);
      CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY,shard INTEGER,row INTEGER);
      CREATE TABLE IF NOT EXISTS stats (kind TEXT,name TEXT,tokens INTEGER,documents INTEGER,
        PRIMARY KEY(kind,name));
      CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY,value TEXT);
    """
    )
    setting = canonical(
        {"identity": identity, "worker": worker, "workers": workers, "code": digest(__file__)}
    )
    saved = db.execute("SELECT value FROM settings WHERE id=1").fetchone()
    if saved and saved[0] != setting:
        raise ValueError("Partition identity changed; use a new worker folder")
    with db:
        db.execute("INSERT OR IGNORE INTO settings VALUES(1,?)", (setting,))
    tokenizer = load_frozen()
    start_shard, start_row = db.execute("SELECT shard,row FROM cursor WHERE id=1").fetchone() or (
        0,
        0,
    )

    def progress():
        rows = db.execute("SELECT kind,name,tokens,documents FROM stats").fetchall()
        return {
            "worker": worker,
            "workers": workers,
            "fresh_code_tokens": sum(t for k, _, t, _ in rows if k == "language"),
            "stats": {
                kind: {n: {"tokens": t, "documents": d} for k, n, t, d in rows if k == kind}
                for kind in ("language", "task")
            },
            "cursor": db.execute("SELECT shard,row FROM cursor WHERE id=1").fetchone(),
            "target": target,
            "updated_unix": time.time(),
        }

    def checkpoint():
        result = progress()
        temporary = folder / "progress.json.partial"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(folder / "progress.json")
        commit()
        return result

    if progress()["fresh_code_tokens"] >= target:
        return checkpoint()
    last_commit = time.monotonic()
    try:
        for shard, name in partition_shards(identity["files"], start_shard, worker, workers):
            downloaded = hf_hub_download(
                CODE_REPO,
                name,
                repo_type="dataset",
                revision=CODE_REVISION,
                local_dir=folder / "download",
            )
            row_index = 0
            for batch in pq.ParquetFile(downloaded).iter_batches(batch_size=256):
                rows = batch.to_pylist()
                begin = row_index
                row_index += len(rows)
                if shard == start_shard and row_index <= start_row:
                    continue
                pending = []
                for offset, row in enumerate(rows):
                    index = begin + offset
                    if shard == start_shard and index < start_row:
                        continue
                    kept = select(row)
                    if kept is None:
                        continue
                    previous = repositories.get(kept["parent"])
                    if previous and previous != "train":
                        continue
                    split = "train" if previous == "train" else split_for(kept["parent"])
                    if split == "test":
                        continue
                    key = hashlib.sha256(kept["content"].encode()).hexdigest()
                    norm = normalized_hash(kept["content"])
                    if (
                        old.execute("SELECT 1 FROM hashes WHERE hash=?", (key,)).fetchone()
                        or old.execute("SELECT 1 FROM normalized WHERE hash=?", (norm,)).fetchone()
                        or db.execute("SELECT 1 FROM dedup WHERE normalized=?", (norm,)).fetchone()
                    ):
                        continue
                    pending.append((key, norm, split, index, kept))
                encodings = tokenizer.encode_batch(
                    [p[4]["content"] for p in pending], add_special_tokens=False
                )
                tokens, documents = Counter(), Counter()
                with db:
                    for (key, norm, split, index, kept), encoded in zip(pending, encodings):
                        if not encoded.ids or any(t < FIRST_CONTENT_TOKEN_ID for t in encoded.ids):
                            continue
                        if not db.execute(
                            "INSERT OR IGNORE INTO dedup VALUES (?,?,?)", (norm, key, split)
                        ).rowcount:
                            continue
                        count = len(encoded.ids) + 1
                        provenance = {
                            "repo": CODE_REPO,
                            "revision": CODE_REVISION,
                            "shard": name,
                            "row": index,
                            **{k: kept[k] for k in ("language", "task_family", "license")},
                            **{k: kept[k] for k in ("repository", "path")},
                        }
                        db.execute(
                            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,0)",
                            (
                                key,
                                "code",
                                split,
                                kept["parent"],
                                canonical(provenance),
                                zlib.compress(kept["content"].encode()),
                                count,
                            ),
                        )
                        if split == "train":
                            for kind in ("language", "task"):
                                value = kept["language" if kind == "language" else "task_family"]
                                tokens[kind, value] += count
                                documents[kind, value] += 1
                    for (kind, value), count in tokens.items():
                        db.execute(
                            "INSERT INTO stats VALUES (?,?,?,?) ON CONFLICT(kind,name) DO UPDATE "
                            "SET tokens=tokens+excluded.tokens,documents=documents+excluded.documents",
                            (kind, value, count, documents[kind, value]),
                        )
                    db.execute("INSERT OR REPLACE INTO cursor VALUES (1,?,?)", (shard, row_index))
                if time.monotonic() - last_commit >= 60:
                    print(json.dumps(checkpoint()), flush=True)
                    last_commit = time.monotonic()
                if progress()["fresh_code_tokens"] >= target:
                    return checkpoint()
            with db:
                db.execute("INSERT OR REPLACE INTO cursor VALUES (1,?,0)", (shard + workers,))
            Path(downloaded).unlink(missing_ok=True)
            checkpoint()
        raise ValueError("Pinned shard partition exhausted before its target")
    finally:
        db.close()
        old.close()
        commit()


def merge_partitions(central, partitions, commit=lambda: None):
    """Merge finished worker databases into `central` with global normalized dedup."""
    db = sqlite3.connect(central)
    db.execute("PRAGMA cache_size=-4194304")  # 4 GiB page cache for the single writer.
    db.executescript(
        SCHEMA
        + """
      CREATE TABLE IF NOT EXISTS stats (kind TEXT,name TEXT,tokens INTEGER,documents INTEGER,
        PRIMARY KEY(kind,name));
      CREATE TABLE IF NOT EXISTS partition_merges (identity TEXT PRIMARY KEY);
      CREATE TABLE IF NOT EXISTS code_norm_rows (normalized TEXT PRIMARY KEY,hash TEXT,split TEXT);
    """
    )
    before = db.execute(
        "SELECT COALESCE(SUM(loss_tokens),0) FROM documents "
        "WHERE source='code' AND split='train' AND conflict=0"
    ).fetchone()[0]
    # Hash-only indexes in memory avoid random reads on a network volume.
    code = {norm: (key, split) for norm, key, split in db.execute("SELECT * FROM code_norm_rows")}
    inserted = 0
    for partition in partitions:
        source = sqlite3.connect(f"file:{Path(partition)}?mode=ro&immutable=1", uri=True)
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
                "SELECT d.*,n.normalized FROM documents d JOIN dedup n ON n.hash=d.hash "
                "ORDER BY d.rowid"
            ):
                *record, norm = row
                previous = code.get(norm)
                if previous:
                    if previous[1] != record[2]:
                        db.execute("UPDATE documents SET conflict=1 WHERE hash=?", (previous[0],))
                    continue
                code[norm] = (record[0], record[2])
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
        for kind, field in (("language", "language"), ("task", "task_family")):
            db.execute(
                f"""INSERT INTO stats SELECT ?,json_extract(provenance,'$.{field}'),
                SUM(loss_tokens),COUNT(*) FROM documents
                WHERE source='code' AND split='train' AND conflict=0
                GROUP BY json_extract(provenance,'$.{field}')""",
                (kind,),
            )
    total = db.execute(
        "SELECT COALESCE(SUM(loss_tokens),0) FROM documents "
        "WHERE source='code' AND split='train' AND conflict=0"
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
