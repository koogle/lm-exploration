"""Disjoint pinned-shard CPU preparation with immutable exclusions and audited merge."""

import hashlib
import json
import shutil
import sqlite3
import time
import zlib
from collections import Counter
from pathlib import Path

from .multilingual_code import CODE_REPO, CODE_REVISION, LANGUAGES, LICENSES, SCHEMA, task_family
from .readiness_audit import normalized_hash
from .simple_llm import SPLIT_VERSION, canonical
from .subword import FIRST_CONTENT_TOKEN_ID, TOKENIZER_SHA256, digest, load_frozen


def partition_shards(files, start, worker, workers):
    if not 0 <= worker < workers or start < 0:
        raise ValueError("Invalid shard partition")
    return [(i, p) for i, p in enumerate(files) if i >= start and i % workers == worker]


def build_exclusions(central, prior, folder, files=None):
    """Called after serial writer is stopped. Keep its committed documents/cursor."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder / "identity.json"
    if metadata.exists():
        return json.loads(metadata.read_text())
    if files is None:
        from huggingface_hub import HfApi

        files = sorted(
            p
            for p in HfApi().list_repo_files(CODE_REPO, repo_type="dataset", revision=CODE_REVISION)
            if p.endswith(".parquet")
        )
    else:
        files = list(files)
    files.sort(key=lambda p: hashlib.sha256(("overnight-code-v1:" + p).encode()).digest())
    temporary = folder / "exclusions.building.sqlite"
    temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    db.execute("ATTACH DATABASE ? AS serial", (str(central),))
    db.execute("ATTACH DATABASE ? AS prior", (str(prior),))
    db.executescript("""
      CREATE TABLE hashes (hash TEXT PRIMARY KEY);
      INSERT INTO hashes SELECT hash FROM serial.prior_hashes;
      INSERT OR IGNORE INTO hashes SELECT hash FROM serial.documents;
      CREATE TABLE normalized (hash TEXT PRIMARY KEY);
      INSERT INTO normalized SELECT hash FROM serial.normalized;
      CREATE TABLE repositories (parent TEXT PRIMARY KEY,split TEXT);
      INSERT INTO repositories SELECT lower(parent),
        CASE WHEN SUM(split!='train')>0 THEN 'heldout' ELSE 'train' END
        FROM prior.documents WHERE source='code_python' GROUP BY lower(parent);
      INSERT INTO repositories SELECT lower(parent),
        CASE WHEN SUM(split!='train')>0 THEN 'heldout' ELSE 'train' END
        FROM serial.documents GROUP BY lower(parent)
        ON CONFLICT(parent) DO UPDATE SET split=CASE WHEN repositories.split!='train'
          OR excluded.split!='train' THEN 'heldout' ELSE 'train' END;
    """)
    cursor = db.execute("SELECT shard,row FROM serial.cursor WHERE id=1").fetchone()
    db.commit()
    db.close()
    temporary.replace(folder / "exclusions.sqlite")
    result = {
        "repo": CODE_REPO,
        "revision": CODE_REVISION,
        "tokenizer": TOKENIZER_SHA256,
        "start_shard": cursor[0],
        "start_row": cursor[1],
        "files": files,
        "exclusions_sha256": digest(folder / "exclusions.sqlite"),
    }
    metadata.write_text(json.dumps(result, indent=2) + "\n")
    return result


def prepare_partition(folder, exclusions, identity, worker, workers, target, commit=lambda: None):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from trainer.loop import atomic_json

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    local = Path("/tmp") / f"code-exclusions-{worker}.sqlite"
    if not local.exists() or digest(local) != identity["exclusions_sha256"]:
        shutil.copyfile(exclusions, local)
    if digest(local) != identity["exclusions_sha256"]:
        raise ValueError("Immutable exclusion ledger checksum mismatch")
    old = sqlite3.connect(f"file:{local}?mode=ro&immutable=1", uri=True)
    repositories = dict(old.execute("SELECT parent,split FROM repositories"))
    db = sqlite3.connect(folder / "raw.sqlite")
    db.execute("PRAGMA cache_size=-262144")
    db.executescript(
        SCHEMA
        + """
      CREATE TABLE IF NOT EXISTS dedup (normalized TEXT PRIMARY KEY,hash TEXT UNIQUE,split TEXT);
      CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY,shard INTEGER,row INTEGER);
      CREATE TABLE IF NOT EXISTS stats (kind TEXT,name TEXT,tokens INTEGER,documents INTEGER,PRIMARY KEY(kind,name));
      CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY,value TEXT);
    """
    )
    setting = canonical(
        {
            "identity": identity,
            "worker": worker,
            "workers": workers,
            "implementation": digest(__file__),
        }
    )
    saved = db.execute("SELECT value FROM settings WHERE id=1").fetchone()
    if saved and saved[0] != setting:
        raise ValueError("Partition identity changed")
    with db:
        db.execute("INSERT OR IGNORE INTO settings VALUES(1,?)", (setting,))
    tokenizer = load_frozen()
    cursor = db.execute("SELECT shard,row FROM cursor WHERE id=1").fetchone()
    start_shard, start_row = cursor or (identity["start_shard"], identity["start_row"])

    def progress():
        rows = db.execute("SELECT kind,name,tokens,documents FROM stats").fetchall()
        return {
            "worker": worker,
            "workers": workers,
            "fresh_code_tokens": sum(t for k, n, t, d in rows if k == "language"),
            "stats": {
                k: {n: {"tokens": t, "documents": d} for kind, n, t, d in rows if kind == k}
                for k in ["language", "task"]
            },
            "cursor": db.execute("SELECT shard,row FROM cursor WHERE id=1").fetchone(),
            "target": target,
            "updated_unix": time.time(),
        }

    def checkpoint():
        result = progress()
        atomic_json(folder / "progress.json", result)
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
                    ri = begin + offset
                    if shard == start_shard and ri < start_row:
                        continue
                    language = str(row.get("language", "")).lower()
                    content = row.get("code")
                    repo = row.get("repo_name")
                    path = row.get("path")
                    if (
                        language not in LANGUAGES
                        or str(row.get("license", "")).lower() not in LICENSES
                        or not repo
                        or not path
                    ):
                        continue
                    if not isinstance(content, str) or not 64 <= len(content) <= 4_000_000:
                        continue
                    if any(
                        "/" + v + "/" in "/" + path.lower()
                        for v in [
                            "node_modules",
                            "vendor",
                            "third_party",
                            "dist",
                            "build",
                            "bower_components",
                        ]
                    ):
                        continue
                    header = content[:1000].lower()
                    if "auto-generated" in header or "automatically generated" in header:
                        continue
                    parent = "code_python:" + repo.lower()
                    previous = repositories.get(parent)
                    if previous and previous != "train":
                        continue
                    bucket = (
                        int(hashlib.sha256(f"{SPLIT_VERSION}:{parent}".encode()).hexdigest(), 16)
                        % 10
                    )
                    split = (
                        "train"
                        if previous == "train"
                        else "validation"
                        if bucket == 0
                        else "test"
                        if bucket == 1
                        else "train"
                    )
                    if split == "test":
                        continue
                    key = hashlib.sha256(content.encode()).hexdigest()
                    norm = normalized_hash(content)
                    if old.execute("SELECT 1 FROM hashes WHERE hash=?", (key,)).fetchone():
                        continue
                    if old.execute("SELECT 1 FROM normalized WHERE hash=?", (norm,)).fetchone():
                        continue
                    if db.execute("SELECT 1 FROM dedup WHERE normalized=?", (norm,)).fetchone():
                        continue
                    pending.append(
                        (
                            key,
                            norm,
                            parent,
                            split,
                            content,
                            language,
                            task_family(path, language),
                            row,
                            ri,
                        )
                    )
                encodings = tokenizer.encode_batch(
                    [p[4] for p in pending], add_special_tokens=False
                )
                additions = Counter()
                documents = Counter()
                with db:
                    for item, encoded in zip(pending, encodings):
                        key, norm, parent, split, content, language, task, row, ri = item
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
                            "row": ri,
                            "language": language,
                            "task_family": task,
                            "license": row["license"],
                            "repository": row["repo_name"],
                            "path": row["path"],
                        }
                        db.execute(
                            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,0)",
                            (
                                key,
                                "code",
                                split,
                                parent,
                                canonical(provenance),
                                zlib.compress(content.encode()),
                                count,
                            ),
                        )
                        if split == "train":
                            for kind, value in [("language", language), ("task", task)]:
                                additions[kind, value] += count
                                documents[kind, value] += 1
                    for (kind, value), count in additions.items():
                        db.execute(
                            "INSERT INTO stats VALUES (?,?,?,?) ON CONFLICT(kind,name) DO UPDATE SET tokens=tokens+excluded.tokens,documents=documents+excluded.documents",
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
        raise ValueError("Pinned shard partition exhausted before fresh target")
    finally:
        db.close()
        old.close()
        commit()


def merge_partitions(central, partitions, commit=lambda: None):
    """Single writer; global whitespace dedup and idempotent shard merge ledger."""
    db = sqlite3.connect(central)
    db.execute("PRAGMA cache_size=-262144")
    db.executescript("""CREATE TABLE IF NOT EXISTS partition_merges (identity TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS code_norm_rows (normalized TEXT PRIMARY KEY,hash TEXT,split TEXT);""")
    before = db.execute(
        "SELECT COALESCE(SUM(loss_tokens),0) FROM documents WHERE source='code' AND split='train' AND conflict=0"
    ).fetchone()[0]
    added = 0
    for partition in partitions:
        source = sqlite3.connect(f"file:{partition}?mode=ro", uri=True)
        identity = hashlib.sha256(
            source.execute("SELECT value FROM settings WHERE id=1").fetchone()[0].encode()
        ).hexdigest()
        cursor = source.execute("SELECT shard,row FROM cursor WHERE id=1").fetchone()
        marker = canonical([identity, cursor])
        if db.execute("SELECT 1 FROM partition_merges WHERE identity=?", (marker,)).fetchone():
            source.close()
            continue
        # Seed lookups only once to preserve original serial-vs-new split exclusions.
        if db.execute("SELECT COUNT(*) FROM code_norm_rows").fetchone()[0] == 0:
            with db:
                for key, blob, split in db.execute(
                    'SELECT hash,text,split FROM documents WHERE source="code"'
                ):
                    db.execute(
                        "INSERT OR IGNORE INTO code_norm_rows VALUES (?,?,?)",
                        (normalized_hash(zlib.decompress(blob).decode()), key, split),
                    )
        with db:
            for row in source.execute(
                "SELECT d.*,n.normalized FROM documents d JOIN dedup n ON n.hash=d.hash"
            ):
                *record, norm = row
                previous = db.execute(
                    "SELECT hash,split FROM code_norm_rows WHERE normalized=?", (norm,)
                ).fetchone()
                if previous:
                    if previous[1] != record[2]:
                        db.execute("UPDATE documents SET conflict=1 WHERE hash=?", (previous[0],))
                    continue
                if db.execute("SELECT 1 FROM normalized WHERE hash=?", (norm,)).fetchone():
                    continue
                db.execute("INSERT INTO normalized VALUES (?)", (norm,))
                db.execute(
                    "INSERT INTO code_norm_rows VALUES (?,?,?)", (norm, record[0], record[2])
                )
                if db.execute(
                    "INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?,?,?)", record
                ).rowcount:
                    added += record[6] if record[2] == "train" else 0
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
        "SELECT SUM(loss_tokens) FROM documents WHERE source='code' AND split='train' AND conflict=0"
    ).fetchone()[0]
    db.close()
    commit()
    return {
        "fresh_code_tokens": total,
        "added_unique_train_tokens": total - before,
        "inserted_train_tokens_before_conflict_exclusions": added,
        "partitions": len(partitions),
        "updated_unix": time.time(),
    }
