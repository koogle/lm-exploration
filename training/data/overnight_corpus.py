"""CPU-only multilingual collection; legacy assembly is not a training-readiness approval."""

import hashlib
import json
import sqlite3
import time
import zlib
from pathlib import Path

from .scholarly import SPLIT_VERSION, canonical
from .subword import FIRST_CONTENT_TOKEN_ID, TOKENIZER_SHA256, digest, load_frozen

CODE_REPO = "codeparrot/github-code-clean"
CODE_REVISION = "c48d40f9e70f0196f8236901ee35807f7d6c44c0"
LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"}
LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "java",
    "c",
    "c++",
    "c#",
    "go",
    "rust",
    "php",
    "ruby",
    "swift",
    "kotlin",
    "scala",
    "sql",
    "shell",
    "bash",
    "powershell",
    "lua",
    "r",
    "julia",
    "haskell",
    "clojure",
    "objective-c",
    "perl",
    "dart",
    "ocaml",
    "elixir",
    "f#",
}
SCHEMA = """CREATE TABLE IF NOT EXISTS documents (
 hash TEXT PRIMARY KEY,source TEXT,split TEXT,parent TEXT,provenance TEXT,
 text BLOB,loss_tokens INTEGER,conflict INTEGER DEFAULT 0);
 CREATE INDEX IF NOT EXISTS document_split ON documents(source,split,conflict,hash);"""


def task_family(path, language):
    value = "/" + path.lower()
    if any(
        x in value for x in ["/test/", "/tests/", "/spec/", "test_", "_test.", ".spec.", ".test."]
    ):
        return "tests"
    if any(
        x in value
        for x in ["/example", "/demo", "/tutorial", "/benchmark", "/algorithm", "/solution"]
    ):
        return "examples_algorithms"
    if language in {"shell", "bash", "powershell"} or any(
        x in value for x in ["/script", "/tools/", "/cli/", "/bin/"]
    ):
        return "tooling"
    return "implementation"


def prepare_code(
    folder, cached_db, eu_db, *, target=1_200_000_000, seconds=6000, commit=lambda: None
):
    """One resumable writer; permissive files, exact and whitespace-normalized dedup."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    from trainer.distributed_train import atomic_json

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "raw.sqlite")
    db.executescript(SCHEMA)
    db.executescript("""CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY,value TEXT);
      CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY,shard INTEGER,row INTEGER);
      CREATE TABLE IF NOT EXISTS prior_hashes (hash TEXT PRIMARY KEY);
      CREATE TABLE IF NOT EXISTS normalized (hash TEXT PRIMARY KEY);
      CREATE TABLE IF NOT EXISTS stats (kind TEXT,name TEXT,tokens INTEGER,documents INTEGER,PRIMARY KEY(kind,name));""")
    identity = canonical(
        {
            "repo": CODE_REPO,
            "revision": CODE_REVISION,
            "tokenizer": TOKENIZER_SHA256,
            "implementation": digest(__file__),
            "cached": str(cached_db),
            "eu": str(eu_db),
        }
    )
    previous = db.execute("SELECT value FROM settings WHERE id=1").fetchone()
    if previous and previous[0] != identity:
        raise ValueError("Multilingual preparation identity changed")
    db.execute("ATTACH DATABASE ? AS prior", (str(cached_db),))
    db.execute("ATTACH DATABASE ? AS oldeu", (str(eu_db),))
    if not previous:
        with db:
            db.execute("INSERT INTO prior_hashes SELECT hash FROM prior.documents")
            db.execute("INSERT OR IGNORE INTO prior_hashes SELECT hash FROM oldeu.documents")
            from .readiness_audit import normalized_hash

            for (blob,) in db.execute(
                "SELECT text FROM prior.documents WHERE source='code_python' AND loss_tokens>0 AND length(text)>0"
            ):
                db.execute(
                    "INSERT OR IGNORE INTO normalized VALUES (?)",
                    (normalized_hash(zlib.decompress(blob).decode()),),
                )
            db.execute("INSERT INTO settings VALUES (1,?)", (identity,))
    # Existing held-out repositories remain held out across languages/case variants.
    repo_splits = {}
    for parent, split in db.execute(
        "SELECT DISTINCT lower(parent),split FROM prior.documents WHERE source='code_python'"
    ):
        key = parent.removeprefix("code_python:")
        if key not in repo_splits or split != "train":
            repo_splits[key] = split
    files = sorted(
        x
        for x in HfApi().list_repo_files(CODE_REPO, repo_type="dataset", revision=CODE_REVISION)
        if x.endswith(".parquet")
    )
    files.sort(key=lambda p: hashlib.sha256(("overnight-code-v1:" + p).encode()).digest())
    tokenizer = load_frozen()
    started = time.monotonic()
    shard, start = db.execute("SELECT shard,row FROM cursor WHERE id=1").fetchone() or (0, 0)

    def status():
        rows = db.execute("SELECT kind,name,tokens,documents FROM stats").fetchall()
        return {
            "fresh_code_tokens": sum(r[2] for r in rows if r[0] == "language"),
            "stats": {
                kind: {n: {"tokens": t, "documents": d} for k, n, t, d in rows if k == kind}
                for kind in ["language", "task"]
            },
            "cursor": db.execute("SELECT shard,row FROM cursor WHERE id=1").fetchone(),
            "repo": CODE_REPO,
            "revision": CODE_REVISION,
            "updated_unix": time.time(),
        }

    try:
        for si in range(shard, len(files)):
            path = hf_hub_download(
                CODE_REPO,
                repo_type="dataset",
                revision=CODE_REVISION,
                filename=files[si],
                local_dir=folder / "download",
            )
            pf = pq.ParquetFile(path)
            row_index = 0
            for batch in pf.iter_batches(batch_size=128):
                rows = batch.to_pylist()
                begin = row_index
                row_index += len(rows)
                if si == shard and row_index <= start:
                    continue
                pending = []
                for offset, row in enumerate(rows):
                    ri = begin + offset
                    if si == shard and ri < start:
                        continue
                    language = str(row.get("language", "")).lower()
                    license_id = str(row.get("license", "")).lower()
                    content = row.get("code")
                    name = row.get("repo_name")
                    file_path = row.get("path")
                    if (
                        language not in LANGUAGES
                        or license_id not in LICENSES
                        or not name
                        or not file_path
                    ):
                        continue
                    if not isinstance(content, str) or not 64 <= len(content) <= 4_000_000:
                        continue
                    if any(
                        "/" + part + "/" in "/" + file_path.lower()
                        for part in [
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
                    canonical_repo = name.lower()
                    prior_split = repo_splits.get(canonical_repo)
                    if prior_split and prior_split != "train":
                        continue
                    parent = "code_python:" + canonical_repo
                    bucket = (
                        int(hashlib.sha256(f"{SPLIT_VERSION}:{parent}".encode()).hexdigest(), 16)
                        % 10
                    )
                    split = (
                        "train"
                        if prior_split == "train"
                        else ("validation" if bucket == 0 else "test" if bucket == 1 else "train")
                    )
                    if split == "test":
                        continue
                    key = hashlib.sha256(content.encode()).hexdigest()
                    if (
                        db.execute("SELECT 1 FROM prior_hashes WHERE hash=?", (key,)).fetchone()
                        or db.execute("SELECT 1 FROM documents WHERE hash=?", (key,)).fetchone()
                    ):
                        continue
                    from .readiness_audit import normalized_hash

                    norm = normalized_hash(content)
                    if db.execute("SELECT 1 FROM normalized WHERE hash=?", (norm,)).fetchone():
                        continue
                    pending.append(
                        (
                            key,
                            parent,
                            split,
                            content,
                            language,
                            task_family(file_path, language),
                            row,
                            ri,
                            norm,
                        )
                    )
                encodings = tokenizer.encode_batch(
                    [p[3] for p in pending], add_special_tokens=False
                )
                with db:
                    for item, encoded in zip(pending, encodings):
                        key, parent, split, content, language, task, row, ri, norm = item
                        if not db.execute(
                            "INSERT OR IGNORE INTO normalized VALUES (?)", (norm,)
                        ).rowcount:
                            continue
                        if not encoded.ids or any(t < FIRST_CONTENT_TOKEN_ID for t in encoded.ids):
                            continue
                        count = len(encoded.ids) + 1
                        provenance = {
                            "repo": CODE_REPO,
                            "revision": CODE_REVISION,
                            "shard": files[si],
                            "row": ri,
                            "language": language,
                            "task_family": task,
                            "license": row["license"],
                            "repository": row["repo_name"],
                            "path": row["path"],
                        }
                        inserted = db.execute(
                            "INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?,?,0)",
                            (
                                key,
                                "code",
                                split,
                                parent,
                                canonical(provenance),
                                zlib.compress(content.encode()),
                                count,
                            ),
                        ).rowcount
                        if inserted and split == "train":
                            for kind, value in [("language", language), ("task", task)]:
                                db.execute(
                                    "INSERT INTO stats VALUES (?,?,?,1) ON CONFLICT(kind,name) DO UPDATE SET tokens=tokens+excluded.tokens,documents=documents+1",
                                    (kind, value, count),
                                )
                    db.execute("INSERT OR REPLACE INTO cursor VALUES (1,?,?)", (si, row_index))
                s = status()
                if row_index % 2048 < 128:
                    atomic_json(folder / "progress.json", s)
                    commit()
                    print(json.dumps(s), flush=True)
                enough = s["fresh_code_tokens"] >= target or time.monotonic() - started >= seconds
                if enough:
                    s["target_met"] = s["fresh_code_tokens"] >= target
                    atomic_json(folder / "progress.json", s)
                    commit()
                    return s
            with db:
                db.execute("INSERT OR REPLACE INTO cursor VALUES (1,?,0)", (si + 1,))
            Path(path).unlink(missing_ok=True)
        raise ValueError("Public multilingual corpus exhausted before minimum fresh capacity")
    finally:
        db.close()
        commit()
