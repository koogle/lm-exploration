"""Merge supplemental documents after writers finish and freeze exact-budget data."""

import gzip
import hashlib
import json
import re
import sqlite3
import zlib
from pathlib import Path

from .python_experiment import TARGETS, Corpus, split_for
from .subword import BOS_TOKEN_ID, EOS_TOKEN_ID, TOKENIZER_SHA256, digest


def merge_documentation(folder, documentation_root, commit=lambda: None):
    corpus = Corpus(folder)
    for complete in sorted(Path(documentation_root).glob("*/complete.json")):
        marker = "documentation:" + complete.parent.name
        if corpus.db.execute("SELECT 1 FROM completed WHERE name=?", (marker,)).fetchone():
            continue
        source = sqlite3.connect(
            f"file:{complete.parent / 'raw.sqlite'}?mode=ro&immutable=1", uri=True
        )
        for key, raw, parent, split, provenance, count in source.execute(
            "SELECT hash,text,parent,split,provenance,loss_tokens FROM documents WHERE conflict=0"
        ):
            text = zlib.decompress(raw).decode("utf-8")
            if hashlib.sha256(text.encode()).hexdigest() != key:
                raise ValueError("Documentation hash mismatch")
            referenced = set(
                re.findall(r"https?://github\.com/([\w.-]+/[\w.-]+)", text, re.IGNORECASE)
            )
            if any(
                split_for(repo.lower()) == "validation"
                or corpus.db.execute(
                    "SELECT 1 FROM excluded_repositories WHERE parent=?", (repo.lower(),)
                ).fetchone()
                for repo in referenced
            ):
                corpus.reject("documentation_references_heldout_repository")
                continue
            corpus.add(
                text,
                "python_documentation",
                parent,
                json.loads(provenance),
                split=split,
                verified_count=count,
            )
        source.close()
        corpus.db.execute("INSERT INTO completed VALUES(?)", (marker,))
        corpus.report("merging_documentation")
        commit()
    result = corpus.report("merged_capacity")
    corpus.db.close()
    commit()
    return result


def freeze(folder, destination):
    folder, destination = Path(folder), Path(destination)
    corpus = Corpus(folder)
    capacity = corpus.report("ready_to_freeze")
    if not capacity["capacity_sufficient"]:
        corpus.db.close()
        raise ValueError(
            "Insufficient measured corpus capacity; mixture and two-pass cap unchanged"
        )
    if destination.exists():
        raise ValueError("Refusing to replace an immutable snapshot")
    destination.mkdir(parents=True)
    database = destination / "raw.sqlite"
    target = sqlite3.connect(database)
    corpus.db.backup(target)
    target.close()
    sources = {}
    for name, budget in TARGETS.items():
        counts = capacity["categories"][name]
        sources[name] = {
            "weight_percent": budget * 100 // 5_000_000_000,
            "target_tokens": budget,
            "max_passes": 1 if counts["unique_train_targets"] >= budget else 2,
            "splits": {
                "train": {"loss_tokens": counts["unique_train_targets"]},
                "validation": {"loss_tokens": counts["validation_targets"]},
            },
        }
    # One fixed window from each of 32 distinct documents per category. Select
    # documents after repository/thread holdout and global exact deduplication.
    windows = 0
    heldout = destination / "heldout.jsonl.gz"
    with gzip.open(heldout, "wt") as writer:
        for name in sources:
            parents = set()
            rows = corpus.db.execute(
                "SELECT hash,text,parent FROM documents WHERE source=? AND split='validation' AND conflict=0 ORDER BY hash",
                (name,),
            )
            selected = []
            for key, raw, parent in rows:
                if parent in parents:
                    continue
                ids = corpus.tokenizer.encode(
                    zlib.decompress(raw).decode(), add_special_tokens=False
                ).ids
                if len(ids) < 32:
                    continue
                parents.add(parent)
                selected.append(
                    {
                        "source": name,
                        "document_hash": key,
                        "tokens": [BOS_TOKEN_ID, *ids, EOS_TOKEN_ID][:8193],
                    }
                )
                if len(selected) == 32:
                    break
            if len(selected) < 8:
                raise ValueError(f"Insufficient representative heldout parents for {name}")
            for row in selected:
                writer.write(json.dumps(row) + "\n")
                windows += 1
    corpus.db.close()
    manifest = {
        "format": "scholarly-raw-v1",
        "training_ready": True,
        "max_passes": 2,
        "experiment": "python-124m-muon-5b",
        "target_tokens": 5_000_000_000,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "database": {"path": "raw.sqlite", "sha256": digest(database)},
        "sources": sources,
        "heldout": {
            "path": heldout.name,
            "sha256": digest(heldout),
            "windows": windows,
            "tokenizer_sha256": TOKENIZER_SHA256,
        },
        "limitations": [
            "Exact file/document deduplication; near clones can remain",
            "Path and content heuristics are imperfect",
            "Documentation includes explanatory Python technical Q&A",
            "Collected recent repositories preferred; supplements include older material",
            "No benchmark-contamination claim; detailed source provenance remains private",
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (destination / "capacity.json").write_text(json.dumps(capacity, indent=2))
    return manifest
