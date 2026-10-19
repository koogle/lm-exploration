"""Read-only corpus audit; derived eligibility and held-out sets never alter old data."""

import gzip
import hashlib
import json
import re
import sqlite3
import time
import zlib
from collections import defaultdict
from pathlib import Path

from .subword import TOKENIZER_SHA256, load_frozen


def normalized_hash(text):
    return hashlib.sha256("".join(text.split()).encode()).hexdigest()


def parent_key(source, parent):
    # Repository identity must survive code-source and case changes.
    if source in {"code", "code_python"}:
        return "repository:" + parent.removeprefix("code_python:").removeprefix("code:").lower()
    return parent


def shingles(text, width=13):
    words = re.findall(r"\w+", text.lower())
    return (tuple(words[i : i + width]) for i in range(max(0, len(words) - width + 1)))


def benchmark_index(folder):
    """Pin public benchmark revisions; failure is fatal, never silently skip checks."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    index = set()
    pins = []
    for repo, fields in [
        ("openai/openai_humaneval", ["prompt", "canonical_solution", "test"]),
        ("google-research-datasets/mbpp", ["text", "code", "test_list"]),
    ]:
        info = HfApi().dataset_info(repo)
        files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))
        if not files:
            raise ValueError(f"No benchmark files: {repo}")
        pins.append({"repo": repo, "revision": info.sha, "files": files})
        for name in files:
            path = hf_hub_download(
                repo,
                name,
                repo_type="dataset",
                revision=info.sha,
                local_dir=Path(folder) / "benchmarks",
            )
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    for field in fields:
                        value = row.get(field, "")
                        text = "\n".join(value) if isinstance(value, list) else str(value)
                        index.update(shingles(text))
    if not index:
        raise ValueError("Empty benchmark fingerprint index")
    return index, pins


def audit(databases, folder, benchmarks, pins, *, commit=lambda: None):
    """Scan all documents, exclude collisions/benchmark parents, build stratified windows.

    databases: [(label, path, fresh_bool)]. Old and new data share one dedup ledger.
    This is conservative lexical screening, not proof against semantic contamination.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "audit.sqlite"
    if path.exists():
        raise ValueError("Audit destination exists; use a new versioned output")
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE records (origin TEXT, hash TEXT, source TEXT, split TEXT, parent TEXT,
          norm TEXT, tokens INTEGER, fresh INTEGER, stratum TEXT, conflict INTEGER,
          benchmark INTEGER, PRIMARY KEY(origin,hash));
        CREATE INDEX normalized ON records(norm);
        CREATE INDEX parents ON records(parent,split);
        CREATE TABLE tokenizer (stratum TEXT, documents INTEGER, tokens INTEGER,
          bytes INTEGER, characters INTEGER, unknowns INTEGER);
    """)
    tokenizer = load_frozen()
    metrics = defaultdict(lambda: [0, 0, 0, 0, 0])
    handles = {}
    scanned = 0
    for label, source_path, fresh in databases:
        source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
        handles[label] = source
        for key, category, split, parent, provenance, blob, count, conflict in source.execute(
            "SELECT hash,source,split,parent,provenance,text,loss_tokens,conflict FROM documents"
        ):
            text = zlib.decompress(blob).decode()
            metadata = json.loads(provenance)
            language = metadata.get("language", "python" if category == "code_python" else "")
            task = metadata.get("task_family", "unclassified")
            stratum = (
                f"code/{language}/{task}"
                if category in {"code", "code_python"}
                else f"{category}/{metadata.get('institution', 'all')}"
            )
            parent = parent_key(category, parent) if parent else "missing:" + key
            hit = any(s in benchmarks for s in shingles(text))
            db.execute(
                "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    label,
                    key,
                    category,
                    split,
                    parent,
                    normalized_hash(text),
                    count,
                    fresh,
                    stratum,
                    conflict,
                    hit,
                ),
            )
            values = metrics[stratum]
            if values[0] < 128:
                sample = text[:32768]
                ids = tokenizer.encode(sample, add_special_tokens=False).ids
                values[0] += 1
                values[1] += len(ids)
                values[2] += len(sample.encode())
                values[3] += len(sample)
                values[4] += ids.count(3)
            scanned += 1
            if scanned % 10000 == 0:
                db.commit()
                (folder / "progress.json").write_text(
                    json.dumps(
                        {
                            "documents_scanned": scanned,
                            "origin": label,
                            "updated_unix": time.time(),
                            "stage": "scanning",
                        }
                    )
                )
                commit()
    db.commit()
    db.executescript("""
      CREATE TABLE blocked_parents AS SELECT DISTINCT parent FROM records
        WHERE benchmark=1 OR conflict=1 OR parent LIKE 'missing:%'
        UNION SELECT parent FROM records GROUP BY parent HAVING COUNT(DISTINCT split)>1;
      CREATE UNIQUE INDEX blocked_index ON blocked_parents(parent);
      CREATE TABLE heldout_norms AS SELECT DISTINCT norm FROM records WHERE split!='train';
      CREATE UNIQUE INDEX heldout_norm_index ON heldout_norms(norm);
      CREATE TABLE eligible AS SELECT * FROM (
        SELECT r.*, ROW_NUMBER() OVER(PARTITION BY norm ORDER BY fresh,origin,hash) AS rank
        FROM records r WHERE NOT EXISTS (SELECT 1 FROM blocked_parents b WHERE b.parent=r.parent)
          AND (split!='train' OR NOT EXISTS(SELECT 1 FROM heldout_norms h WHERE h.norm=r.norm))
      ) WHERE rank=1;
      CREATE INDEX eligible_split ON eligible(split,stratum,parent);
    """)
    capacities = [
        dict(zip(["source", "fresh", "documents", "unique_train_tokens"], row))
        for row in db.execute(
            "SELECT source,fresh,COUNT(*),SUM(tokens) FROM eligible WHERE split='train' GROUP BY source,fresh"
        )
    ]
    # Uniform families per stratum, deterministic document and offsets within family.
    selected = defaultdict(dict)
    for origin, key, parent, count, stratum in db.execute(
        "SELECT origin,hash,parent,tokens,stratum FROM eligible WHERE split='validation' ORDER BY hash"
    ):
        families = selected[stratum]
        rank = hashlib.sha256(parent.encode()).hexdigest()
        if parent not in families:
            families[parent] = (rank, origin, key, count)
        if len(families) > 256:
            del families[max(families, key=lambda p: families[p][0])]
    coverage = {}
    with gzip.open(folder / "heldout-windows.jsonl.gz", "wt") as output:
        for stratum, families in sorted(selected.items()):
            targets = windows = 0
            for parent, (_, origin, key, _) in families.items():
                blob = (
                    handles[origin]
                    .execute("SELECT text FROM documents WHERE hash=?", (key,))
                    .fetchone()[0]
                )
                ids = tokenizer.encode(
                    zlib.decompress(blob).decode(), add_special_tokens=False
                ).ids + [2]
                offsets = sorted({0, max(0, len(ids) - 2049)})
                for offset in offsets:
                    tokens = [1] + ids[offset : offset + 2048]
                    output.write(
                        json.dumps(
                            {
                                "stratum": stratum,
                                "parent": parent,
                                "hash": key,
                                "offset": offset,
                                "tokens": tokens,
                            }
                        )
                        + "\n"
                    )
                    targets += len(tokens) - 1
                    windows += 1
            coverage[stratum] = {"parents": len(families), "windows": windows, "targets": targets}
    for stratum, values in metrics.items():
        db.execute("INSERT INTO tokenizer VALUES (?,?,?,?,?,?)", (stratum, *values))
    result = {
        "stage": "audited_not_training_ready",
        "tokenizer_sha256": TOKENIZER_SHA256,
        "databases": [(n, str(p), f) for n, p, f in databases],
        "benchmarks": pins,
        "capacity": capacities,
        "heldout": coverage,
        "tokenizer": {
            s: dict(zip(["documents", "tokens", "utf8_bytes", "characters", "unknowns"], v))
            for s, v in metrics.items()
        },
        "documents_scanned": scanned,
        "benchmark_hit_documents": db.execute(
            "SELECT COUNT(*) FROM records WHERE benchmark=1"
        ).fetchone()[0],
        "blocked_parents": db.execute("SELECT COUNT(*) FROM blocked_parents").fetchone()[0],
        "pending": [
            "HumanEval/MBPP lexical screen does not cover other benchmarks or semantic overlap",
            "Review language/task balance and tokenizer efficiency before selecting the mix",
            "Run representative held-out evaluation in trainer before any substantial run",
            "Existing historical training contamination cannot be undone by future filtering",
        ],
    }
    db.commit()
    db.close()
    for source in handles.values():
        source.close()
    (folder / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    commit()
    return result
