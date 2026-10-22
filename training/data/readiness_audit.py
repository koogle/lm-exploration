"""Read-only training-readiness audit across one or more raw corpus databases.

The audit never changes its inputs. It scans every document into `audit.sqlite` and
derives:
- whitespace-normalized dedup across all databases (prior copies win over fresh ones);
- blocked parents: benchmark hits (13-word HumanEval/MBPP shingles), conflicts, missing
  parents and parents that appear in more than one split;
- train documents whose text or hash also appears in a held-out split;
- eligible training capacity per source, and a stratified frozen held-out set
  (`heldout-windows.jsonl.gz`, at most 256 parents per stratum, in the format
  `heldout.py` loads).
This is conservative lexical screening, not proof against semantic contamination.
"""

import gzip
import hashlib
import json
import sqlite3
import time
import zlib
from collections import defaultdict
from pathlib import Path

from .lexical_screen import LexicalScreen, normalized_hash, shingles
from .subword import BOS_TOKEN_ID, EOS_TOKEN_ID, TOKENIZER_SHA256, UNK_TOKEN_ID, load_frozen

CODE_SOURCES = {"code", "code_python"}
BENCHMARKS = [
    ("openai/openai_humaneval", ["prompt", "canonical_solution", "test"]),
    ("google-research-datasets/mbpp", ["text", "code", "test_list"]),
]


def parent_key(source, parent):
    """Code repositories are identified across code sources and letter case."""
    if source in CODE_SOURCES:
        return "repository:" + parent.split(":", 1)[-1].lower()
    return parent


def benchmark_index(folder):
    """13-word shingles of pinned public benchmarks; a missing benchmark is fatal."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    index, pins = set(), []
    for repo, fields in BENCHMARKS:
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


def audit(
    databases,
    folder,
    benchmarks,
    pins,
    *,
    context=2048,
    commit=lambda: None,
    benchmark_screen=True,
    tokenizer_diagnostics=True,
    resume=False,
):
    """Audit `databases`, a list of `(label, path, fresh)`; write `report.json`.

    `benchmarks` and `pins` come from `benchmark_index`. With `resume=True` the scan
    continues from committed metadata and the derived tables are rebuilt, so screening
    options can change between runs.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "audit.sqlite"
    if path.exists() and not resume:
        raise ValueError("Audit destination exists; use a new versioned output")
    db = sqlite3.connect(path)
    db.execute("PRAGMA cache_size=-4194304")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS records (origin TEXT, hash TEXT, source TEXT, split TEXT,
          parent TEXT, norm TEXT, tokens INTEGER, fresh INTEGER, stratum TEXT, conflict INTEGER,
          benchmark INTEGER, PRIMARY KEY(origin,hash));
        CREATE INDEX IF NOT EXISTS normalized ON records(norm);
        CREATE INDEX IF NOT EXISTS parents ON records(parent,split);
        DROP TABLE IF EXISTS tokenizer;
        CREATE TABLE tokenizer (stratum TEXT, documents INTEGER, tokens INTEGER,
          bytes INTEGER, characters INTEGER, unknowns INTEGER);
    """)
    tokenizer = load_frozen()
    screen = LexicalScreen(benchmarks) if benchmark_screen else None
    metrics = defaultdict(lambda: [0, 0, 0, 0, 0])
    handles = {}
    scanned = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    for label, source_path, fresh in databases:
        source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
        handles[label] = source
        # Skip already-scanned rows in SQL, so their text pages are never read.
        seen = {row[0] for row in db.execute("SELECT hash FROM records WHERE origin=?", (label,))}
        source.create_function("already_scanned", 1, seen.__contains__)
        for key, category, split, parent, provenance, blob, count, conflict in source.execute(
            "SELECT hash,source,split,parent,provenance,text,loss_tokens,conflict "
            "FROM documents d WHERE NOT already_scanned(d.hash)"
        ):
            # Test records may keep only their hash and parent, to protect split identity.
            tombstone = count == 0 and not blob
            text = "" if tombstone else zlib.decompress(blob).decode()
            metadata = json.loads(provenance)
            language = metadata.get("language", "python" if category == "code_python" else "")
            if category in CODE_SOURCES:
                stratum = f"code/{language}/{metadata.get('task_family', 'unclassified')}"
            else:
                stratum = f"{category}/{metadata.get('institution', 'all')}"
            db.execute(
                "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    label,
                    key,
                    category,
                    split,
                    parent_key(category, parent) if parent else "missing:" + key,
                    "excluded:" + key if tombstone else normalized_hash(text),
                    count,
                    fresh,
                    stratum,
                    conflict,
                    screen.matches(text) if screen else False,
                ),
            )
            values = metrics[stratum] if tokenizer_diagnostics else None
            if values is not None and not tombstone and values[0] < 128:
                sample = text[:32768]
                ids = tokenizer.encode(sample, add_special_tokens=False).ids
                values[0] += 1
                values[1] += len(ids)
                values[2] += len(sample.encode())
                values[3] += len(sample)
                values[4] += ids.count(UNK_TOKEN_ID)
            scanned += 1
            if scanned % 10000 == 0:
                db.commit()
                (folder / "progress.json").write_text(
                    json.dumps(
                        {"documents_scanned": scanned, "origin": label, "updated_unix": time.time()}
                    )
                )
                commit()
    db.commit()
    db.executescript(
        """
      DROP TABLE IF EXISTS blocked_parents;
      DROP TABLE IF EXISTS heldout_norms;
      DROP TABLE IF EXISTS heldout_hashes;
      DROP TABLE IF EXISTS eligible;
      CREATE TABLE blocked_parents AS SELECT DISTINCT parent FROM records
        WHERE (benchmark=1 AND SCREEN_BENCHMARKS) OR conflict=1 OR parent LIKE 'missing:%'
        UNION SELECT parent FROM records GROUP BY parent HAVING COUNT(DISTINCT split)>1;
      CREATE UNIQUE INDEX blocked_index ON blocked_parents(parent);
      CREATE TABLE heldout_norms AS SELECT DISTINCT norm FROM records WHERE split!='train';
      CREATE UNIQUE INDEX heldout_norm_index ON heldout_norms(norm);
      CREATE TABLE heldout_hashes AS SELECT DISTINCT hash FROM records WHERE split!='train';
      CREATE UNIQUE INDEX heldout_hash_index ON heldout_hashes(hash);
      CREATE TABLE eligible AS SELECT * FROM (
        SELECT r.*, ROW_NUMBER() OVER(PARTITION BY norm ORDER BY fresh,origin,hash) AS rank
        FROM records r WHERE NOT EXISTS (SELECT 1 FROM blocked_parents b WHERE b.parent=r.parent)
          AND tokens>0 AND (split!='train' OR (
            NOT EXISTS(SELECT 1 FROM heldout_norms h WHERE h.norm=r.norm)
            AND NOT EXISTS(SELECT 1 FROM heldout_hashes h WHERE h.hash=r.hash)))
      ) WHERE rank=1;
      CREATE INDEX eligible_split ON eligible(split,stratum,parent);
    """.replace("SCREEN_BENCHMARKS", "1" if benchmark_screen else "0")
    )
    capacities = [
        dict(zip(["source", "fresh", "documents", "unique_train_tokens"], row))
        for row in db.execute(
            "SELECT source,fresh,COUNT(*),SUM(tokens) FROM eligible WHERE split='train' "
            "GROUP BY source,fresh"
        )
    ]
    # One document per parent, up to 256 parents per stratum, chosen by parent hash.
    selected = defaultdict(dict)
    for origin, key, source, parent, stratum in db.execute(
        "SELECT origin,hash,source,parent,stratum FROM eligible WHERE split='validation' "
        "ORDER BY hash"
    ):
        families = selected[stratum]
        if parent not in families:
            families[parent] = (hashlib.sha256(parent.encode()).hexdigest(), origin, key, source)
        if len(families) > 256:
            del families[max(families, key=lambda p: families[p][0])]
    coverage = {}
    heldout_path = folder / "heldout-windows.jsonl.gz"
    with gzip.open(heldout_path, "wt") as output:
        for stratum, families in sorted(selected.items()):
            targets = windows = 0
            for parent, (_, origin, key, source) in families.items():
                blob = (
                    handles[origin]
                    .execute("SELECT text FROM documents WHERE hash=?", (key,))
                    .fetchone()[0]
                )
                ids = tokenizer.encode(zlib.decompress(blob).decode(), add_special_tokens=False).ids
                sequence = [BOS_TOKEN_ID, *ids, EOS_TOKEN_ID]
                # The first window and, for long documents, the last one.
                for offset in sorted({0, max(0, len(sequence) - context - 1)}):
                    tokens = sequence[offset : offset + context + 1]
                    row = {
                        "source": source,
                        "stratum": stratum,
                        "parent": parent,
                        "hash": key,
                        "offset": offset,
                        "tokens": tokens,
                    }
                    output.write(json.dumps(row) + "\n")
                    targets += len(tokens) - 1
                    windows += 1
            coverage[stratum] = {"parents": len(families), "windows": windows, "targets": targets}
    for stratum, values in metrics.items():
        db.execute("INSERT INTO tokenizer VALUES (?,?,?,?,?,?)", (stratum, *values))
    names = ["documents", "tokens", "utf8_bytes", "characters", "unknowns"]
    result = {
        "stage": "audited_not_training_ready",
        "tokenizer_sha256": TOKENIZER_SHA256,
        "databases": [(label, str(p), fresh) for label, p, fresh in databases],
        "benchmarks": pins if benchmark_screen else [],
        "policy": {
            "benchmark_screen": benchmark_screen,
            "tokenizer_diagnostics": tokenizer_diagnostics,
            "validation_exclusion": "parent and exact/whitespace-normalized document identity",
        },
        "capacity": capacities,
        "heldout": coverage,
        "heldout_windows": {
            "path": heldout_path.name,
            "windows": sum(c["windows"] for c in coverage.values()),
            "context": context,
        },
        "tokenizer": {s: dict(zip(names, v)) for s, v in metrics.items()},
        "documents_scanned": scanned,
        "benchmark_hit_documents": (
            db.execute("SELECT COUNT(*) FROM records WHERE benchmark=1").fetchone()[0]
            if benchmark_screen
            else None
        ),
        "blocked_parents": db.execute("SELECT COUNT(*) FROM blocked_parents").fetchone()[0],
        "pending": [
            (
                "HumanEval/MBPP lexical screen does not cover other benchmarks or semantic overlap"
                if benchmark_screen
                else "Benchmark contamination screening skipped"
            ),
            "Review language/task balance and tokenizer efficiency before selecting the mix",
        ],
    }
    db.commit()
    db.close()
    for source in handles.values():
        source.close()
    (folder / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    commit()
    return result
