"""Reference completed, immutable preparation databases with sampled held-outs and no global audit."""

import gzip
import json
import sqlite3
import zlib
from fractions import Fraction
from pathlib import Path

from .multilingual_code import SCHEMA
from .subword import TOKENIZER_SHA256, digest, load_frozen


def literal(value):
    return "'" + value.replace("'", "''") + "'"


def attach_prepared(db, manifest):
    """Read immutable completed corpora through a small, checkpointed manifest."""
    db.execute("CREATE TEMP TABLE excluded_hash(hash TEXT PRIMARY KEY)")
    db.execute("CREATE TEMP TABLE excluded_parent(parent TEXT PRIMARY KEY)")
    db.executemany(
        "INSERT OR IGNORE INTO excluded_hash VALUES(?)",
        [(h,) for h in manifest["excluded_validation_hashes"]],
    )
    db.executemany(
        "INSERT OR IGNORE INTO excluded_parent VALUES(?)",
        [(p,) for p in manifest["excluded_validation_parents"]],
    )
    views = []
    for i, artifact in enumerate(manifest["external_databases"]):
        path = Path(artifact["path"])
        if not path.is_absolute() or path.stat().st_size != artifact["size_bytes"]:
            raise ValueError("Prepared database path/size changed")
        alias = f"prepared_{i}"
        db.execute(f"ATTACH DATABASE ? AS {alias}", (path.as_uri() + "?mode=ro&immutable=1",))
        for original, source in artifact["sources"].items():
            views.append(
                f"SELECT hash,{literal(source)} AS source,split,parent,provenance,text,loss_tokens,conflict FROM {alias}.documents WHERE source={literal(original)} AND conflict=0 AND NOT EXISTS(SELECT 1 FROM excluded_hash e WHERE e.hash=documents.hash) AND NOT EXISTS(SELECT 1 FROM excluded_parent e WHERE e.parent=documents.parent)"
            )
    db.execute("CREATE TEMP VIEW documents AS " + " UNION ALL ".join(views))


def build(destination, weights, databases, train_counts, *, target, pass_caps=None):
    """Freeze references, not another multi-GB copy of already prepared text.

    `weights` maps each mixture source to its percentage (summing to 100).
    `databases` lists `(path, {stored_source: mixture_source})` for completed,
    immutable preparation databases. `train_counts` gives each mixture source's
    prepared training targets; `pass_caps` gives per-source pass caps (default 2).
    Up to 16 validation parents per stored source become the frozen held-out set,
    and their hashes/parents are excluded from training reads.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise ValueError("Refuse to overwrite a prepared snapshot")
    weights = {name: Fraction(str(value)) for name, value in weights.items()}
    if sum(weights.values()) != 100 or set(train_counts) != set(weights):
        raise ValueError("Expected normalized weights and a train count for every source")
    pass_caps = {name: 2 for name in weights} | dict(pass_caps or {})
    if any(cap not in (1, 2) for cap in pass_caps.values()):
        raise ValueError("Source pass cap must be one or two")
    tokenizer = load_frozen()
    artifacts = []
    heldout, parents, hashes = [], set(), set()
    for path, mapping in databases:
        if set(mapping.values()) - set(weights):
            raise ValueError("Database maps to a source outside the mixture")
        path = Path(path).resolve()
        artifacts.append(
            {"path": str(path), "size_bytes": path.stat().st_size, "sources": dict(mapping)}
        )
        db = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
        for original_name, source in mapping.items():
            selected = set()
            for key, parent, provenance, blob in db.execute(
                "SELECT hash,parent,provenance,text FROM documents WHERE source=? AND split='validation' AND conflict=0 LIMIT 256",
                (original_name,),
            ):
                if parent in selected or len(selected) >= 16:
                    continue
                selected.add(parent)
                parents.add(parent)
                hashes.add(key)
                text = zlib.decompress(blob).decode()
                ids = tokenizer.encode(text[:32768], add_special_tokens=False).ids
                if not ids:
                    continue
                tokens = [1, *ids, *([2] if len(text) <= 32768 else [])][:2049]
                metadata = json.loads(provenance)
                heldout.append(
                    {
                        "source": source,
                        "stratum": f"{source}/{metadata.get('language', 'all')}/{metadata.get('task_family', 'sample')}",
                        "parent": parent,
                        "hash": key,
                        "tokens": tokens,
                    }
                )
        db.close()
    sources = {}
    for name, weight in weights.items():
        validation = sum(len(row["tokens"]) - 1 for row in heldout if row["source"] == name)
        count = train_counts[name]
        cap = pass_caps[name]
        if not validation or count * cap < target * float(weight) / 100:
            raise ValueError(
                f"Prepared capacity/validation missing for {name}: {count}, {validation}"
            )
        sources[name] = {
            "weight_percent": float(weight),
            "weight_percent_exact": str(weight),
            "max_passes": cap,
            "splits": {"train": {"loss_tokens": count}, "validation": {"loss_tokens": validation}},
        }
    path = destination / "raw.sqlite"
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.close()
    with gzip.open(destination / "heldout.jsonl.gz", "wt") as out:
        for row in heldout:
            out.write(json.dumps(row) + "\n")
    manifest = {
        "format": "scholarly-raw-v1",
        "training_ready": False,
        "max_passes": 2,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "database": {"path": "raw.sqlite", "sha256": digest(path)},
        "external_databases": artifacts,
        "sources": sources,
        "excluded_validation_hashes": sorted(hashes),
        "excluded_validation_parents": sorted(parents),
        "heldout": {
            "path": "heldout.jsonl.gz",
            "sha256": digest(destination / "heldout.jsonl.gz"),
            "tokenizer_sha256": TOKENIZER_SHA256,
            "windows": len(heldout),
        },
        "phase_target_tokens": target,
        "audit_policy": {
            "global_audit": False,
            "benchmark_screen": False,
            "tokenizer_diagnostics": False,
            "validation": "sampled parents and exact document hashes excluded",
        },
        "pending": [
            "No global corpus audit",
            "External completed databases checked by path/size, not full-file checksums",
            "Capacity uses caller-supplied preparation counters before sampled-parent exclusions",
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
