"""Snapshots that reference completed preparation databases instead of copying them.

`build` freezes a small manifest over immutable `raw.sqlite` databases (for example
several finished corpus preparations), maps their stored source names onto mixture
sources and samples a frozen held-out set from their validation splits. Training reads
the databases through `attach_prepared`, which hides every sampled held-out document and
every document sharing a held-out parent. There is no global audit: external databases
are checked by path and size, not full checksums.
"""

import itertools
import json
import sqlite3
import zlib
from fractions import Fraction
from pathlib import Path

from .heldout import write_heldout
from .subword import BOS_TOKEN_ID, EOS_TOKEN_ID, TOKENIZER_SHA256, digest, load_frozen

DOCUMENTS = """CREATE TABLE documents (
    hash TEXT PRIMARY KEY, source TEXT, split TEXT, parent TEXT,
    provenance TEXT, text BLOB, loss_tokens INTEGER, conflict INTEGER DEFAULT 0)"""


def _literal(value):
    return "'" + value.replace("'", "''") + "'"


def attach_prepared(db, manifest):
    """Expose the referenced databases as temporary views on `db`.

    Each (database, stored source) pair becomes a view `part_{n}` whose `ordinal` is the
    stored row ID, so it reads in insertion order without sorting; `documents` is their
    union. Returns `{mixture source: [view names in reading order]}`.
    """
    db.execute("CREATE TEMP TABLE excluded_hash(hash TEXT PRIMARY KEY)")
    db.execute("CREATE TEMP TABLE excluded_parent(parent TEXT PRIMARY KEY)")
    db.executemany(
        "INSERT INTO excluded_hash VALUES(?)", [(h,) for h in manifest["excluded_hashes"]]
    )
    db.executemany(
        "INSERT INTO excluded_parent VALUES(?)", [(p,) for p in manifest["excluded_parents"]]
    )
    parts, numbers = {}, itertools.count()
    for index, artifact in enumerate(manifest["external_databases"]):
        path = Path(artifact["path"])
        if not path.is_absolute() or path.stat().st_size != artifact["size_bytes"]:
            raise ValueError("Prepared database path or size changed")
        alias = f"prepared_{index}"
        db.execute(f"ATTACH DATABASE ? AS {alias}", (path.as_uri() + "?mode=ro&immutable=1",))
        for stored, source in artifact["sources"].items():
            view = f"part_{next(numbers)}"
            db.execute(
                f"CREATE TEMP VIEW {view} AS SELECT d.rowid AS ordinal,hash,"
                f"{_literal(source)} AS source,split,parent,provenance,text,loss_tokens,conflict "
                f"FROM {alias}.documents d WHERE source={_literal(stored)} AND conflict=0 "
                "AND NOT EXISTS(SELECT 1 FROM excluded_hash e WHERE e.hash=d.hash) "
                "AND NOT EXISTS(SELECT 1 FROM excluded_parent e WHERE e.parent=d.parent)"
            )
            parts.setdefault(source, []).append(view)
    views = [view for names in parts.values() for view in names]
    db.execute(
        "CREATE TEMP VIEW documents AS " + " UNION ALL ".join(f"SELECT * FROM {v}" for v in views)
    )
    return parts


def _sample_heldout(path, mapping, tokenizer, context, parents_per_source):
    """Up to `parents_per_source` validation parents per stored source, hash ordered."""
    db = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    rows = []
    try:
        for stored, source in mapping.items():
            selected = set()
            for key, parent, provenance, blob in db.execute(
                "SELECT hash,parent,provenance,text FROM documents "
                "WHERE source=? AND split='validation' AND conflict=0 ORDER BY hash",
                (stored,),
            ):
                if len(selected) >= parents_per_source:
                    break
                if parent in selected:
                    continue
                selected.add(parent)
                text = zlib.decompress(blob).decode()
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                metadata = json.loads(provenance)
                language = metadata.get("language", "all")
                task = metadata.get("task_family", "sample")
                rows.append(
                    {
                        "source": source,
                        "stratum": f"{source}/{language}/{task}",
                        "parent": parent,
                        "hash": key,
                        "tokens": [BOS_TOKEN_ID, *ids, EOS_TOKEN_ID][: context + 1],
                    }
                )
    finally:
        db.close()
    return rows


def build(destination, weights, databases, *, target, context=2048, parents_per_source=16):
    """Freeze a referencing snapshot for `target` training tokens.

    `weights` maps each mixture source to its percentage (summing to 100). `databases`
    lists `(path, {stored_source: mixture_source})` for completed, immutable databases.
    Training capacity is counted after held-out exclusions and must cover `target`
    within two passes for every source.
    """
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Refuse to overwrite a prepared snapshot")
    weights = {name: Fraction(str(value)) for name, value in weights.items()}
    if sum(weights.values()) != 100:
        raise ValueError("Weights must sum to 100")
    tokenizer = load_frozen()
    artifacts, heldout = [], []
    for path, mapping in databases:
        if set(mapping.values()) - set(weights):
            raise ValueError("Database maps to a source outside the mixture")
        path = Path(path).resolve()
        artifacts.append(
            {"path": str(path), "size_bytes": path.stat().st_size, "sources": dict(mapping)}
        )
        heldout += _sample_heldout(path, mapping, tokenizer, context, parents_per_source)
    references = {
        "external_databases": artifacts,
        "excluded_hashes": sorted({row["hash"] for row in heldout}),
        "excluded_parents": sorted({row["parent"] for row in heldout}),
    }
    # Count exactly what training will read: the same view, after exclusions.
    counter = sqlite3.connect(":memory:", uri=True)  # ATTACH takes read-only URIs.
    try:
        attach_prepared(counter, references)
        train = dict(
            counter.execute(
                "SELECT source,SUM(loss_tokens) FROM documents WHERE split='train' GROUP BY source"
            )
        )
        # A document in two referenced databases would be trained twice per pass.
        repeated = counter.execute("SELECT COUNT(*)-COUNT(DISTINCT hash) FROM documents")
        if repeated.fetchone()[0]:
            raise ValueError("Referenced databases share documents; deduplicate them first")
    finally:
        counter.close()
    sources = {}
    for name, weight in weights.items():
        validation = sum(len(row["tokens"]) - 1 for row in heldout if row["source"] == name)
        count = train.get(name, 0)
        if not validation or 2 * count < target * weight / 100:
            raise ValueError(f"Source {name} lacks held-out data or two-pass capacity")
        sources[name] = {
            "weight_percent": float(weight),
            "weight_percent_exact": str(weight),
            "splits": {"train": {"loss_tokens": count}, "validation": {"loss_tokens": validation}},
        }
    destination.mkdir(parents=True)
    database = destination / "raw.sqlite"
    db = sqlite3.connect(database)
    db.execute(DOCUMENTS)
    db.commit()
    db.close()
    manifest = {
        "format": "raw-text-v1",
        "training_ready": False,
        "max_passes": 2,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "database": {"path": "raw.sqlite", "sha256": digest(database)},
        **references,
        "sources": sources,
        "heldout": write_heldout(destination, heldout),
        "target_tokens": target,
        "pending": [
            "No global corpus audit; held-out documents and parents sampled per source",
            "External databases are checked by path and size, not full-file checksums",
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", required=True, help="JSON: weights, databases, target, context")
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    result = build(
        args.destination,
        spec["weights"],
        [(entry["path"], entry["sources"]) for entry in spec["databases"]],
        target=spec["target"],
        context=spec.get("context", 2048),
    )
    print(json.dumps({name: s["splits"] for name, s in result["sources"].items()}, indent=2))
