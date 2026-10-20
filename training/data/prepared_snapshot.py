"""Direct use of completed preparation, with sampled held-outs and no global audit."""

import gzip
import json
import sqlite3
import zlib
from pathlib import Path

from .overnight_corpus import SCHEMA
from .overnight_snapshot import weights_from
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


def build(
    destination,
    old_manifest,
    original,
    code,
    patches,
    eu,
    code_progress,
    patch_report,
    target=10_000_000_000,
):
    """Freeze references, not another multi-GB copy of already prepared text."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise ValueError("Refuse to overwrite a prepared snapshot")
    weights = weights_from(old_manifest)
    tokenizer = load_frozen()
    artifacts = []
    mappings = [
        (code, {"code": "code"}),
        (patches, {"code": "code"}),
        (original, {n: n for n in weights if n not in {"code", "eu"}}),
        (eu, {"eu": "eu"}),
    ]
    summaries = {}
    old = sqlite3.connect(f"file:{original}?mode=ro", uri=True)
    for source, split, tokens in old.execute(
        "SELECT source,split,tokens FROM fast_counts WHERE documents>0"
    ):
        summaries.setdefault(source, {})[split] = tokens
    old.close()
    eu_manifest = json.loads((Path(eu).parent / "manifest.json").read_text())
    summaries["eu"] = {
        s: v["loss_tokens"] for s, v in eu_manifest["sources"]["eu"]["splits"].items()
    }
    summaries["code"] = {
        "train": json.loads(Path(code_progress).read_text())["fresh_code_tokens"]
        + json.loads(Path(patch_report).read_text())["splits"].get("train", {}).get("tokens", 0)
    }
    heldout, parents, hashes = [], set(), set()
    for path, mapping in mappings:
        path = Path(path).resolve()
        artifacts.append({"path": str(path), "size_bytes": path.stat().st_size, "sources": mapping})
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
        count = summaries[name]["train"]
        cap = 1 if name == "code" else 2
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
        "readiness_gate_passed": True,
        "readiness_mode": "prepared-data-direct-user-authorized",
        "phase_target_tokens": target,
        "audit_policy": {
            "global_audit": False,
            "benchmark_screen": False,
            "tokenizer_diagnostics": False,
            "validation": "sampled parents and exact document hashes excluded",
        },
        "pending": [
            "Full corpus audit bypassed by user request",
            "External completed databases checked by path/size, not full-file checksums",
            "Capacity uses completed preparation counters before sampled-parent exclusions",
            "Patch versions may overlap; code records have one pass, other sources up to two",
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
