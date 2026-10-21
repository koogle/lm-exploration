"""Merge supplemental documents after writers finish and freeze exact-budget data."""

import gzip
import hashlib
import json
import re
import sqlite3
import zlib
from pathlib import Path

from .python_documentation import python2_specific
from .python_experiment import TARGETS, Corpus, placeholder_scaffold, split_for
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
            meta = json.loads(provenance)
            if python2_specific(text):
                corpus.reject("python2_specific_documentation")
                continue
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
                meta,
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
    target.execute(
        "CREATE INDEX IF NOT EXISTS collected_priority ON documents(source,split,conflict,"
        "(CASE WHEN json_extract(provenance,'$.origin')='collected_git' THEN 0 ELSE 1 END),hash)"
    )
    target.commit()
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
    capacity["by_origin"] = [
        {"category": name, "origin": origin, "documents": count, "unique_train_targets": tokens}
        for name, origin, count, tokens in corpus.db.execute(
            "SELECT source,json_extract(provenance,'$.origin'),COUNT(*),SUM(loss_tokens) "
            "FROM documents WHERE split='train' AND conflict=0 "
            "GROUP BY source,json_extract(provenance,'$.origin') ORDER BY source,2"
        )
    ]
    corpus.db.close()
    manifest = {
        "format": "scholarly-raw-v1",
        "training_ready": True,
        "max_passes": 2,
        "experiment": "python-124m-muon-5b",
        "prefer_collected": True,
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
    (destination / "capacity.md").write_text((folder / "capacity.md").read_text())
    return manifest


def merge_code(folder, supplemental_root, commit=lambda: None):
    """Preserve collected repository versions and deduplicate the larger code pool."""
    corpus = Corpus(folder)
    for complete in sorted(Path(supplemental_root).glob("*/complete.json")):
        marker = "full_python_code:" + complete.parent.name
        if corpus.db.execute("SELECT 1 FROM completed WHERE name=?", (marker,)).fetchone():
            continue
        source = sqlite3.connect(
            f"file:{complete.parent / 'raw.sqlite'}?mode=ro&immutable=1", uri=True
        )
        for key, raw, name, parent, split, provenance, count in source.execute(
            "SELECT hash,text,source,parent,split,provenance,loss_tokens FROM documents WHERE conflict=0"
        ):
            if name not in ("python_implementation", "python_tests_examples"):
                raise ValueError("Unexpected supplemental code category")
            if corpus.db.execute("SELECT 1 FROM repositories WHERE parent=?", (parent,)).fetchone():
                corpus.reject("older_collected_repository_version")
                continue
            text = zlib.decompress(raw).decode("utf-8")
            if hashlib.sha256(text.encode()).hexdigest() != key:
                raise ValueError("Supplemental code hash mismatch")
            if placeholder_scaffold(text):
                corpus.reject("placeholder_test_scaffold")
                continue
            corpus.add(
                text, name, parent, json.loads(provenance), split=split, verified_count=count
            )
        source.close()
        corpus.db.execute("INSERT INTO completed VALUES(?)", (marker,))
        corpus.report("merging_full_python_code")
        commit()
    result = corpus.report("merged_code_capacity")
    corpus.db.close()
    commit()
    return result


def exclude_existing_scaffolds(folder, commit=lambda: None):
    corpus = Corpus(folder)
    marker = "quality:placeholder_scaffolds_v1"
    if not corpus.db.execute("SELECT 1 FROM completed WHERE name=?", (marker,)).fetchone():
        rejected = [
            (key,)
            for key, raw in corpus.db.execute(
                "SELECT hash,text FROM documents WHERE source IN ('python_implementation','python_tests_examples') AND conflict=0"
            )
            if placeholder_scaffold(zlib.decompress(raw).decode("utf-8"))
        ]
        corpus.db.executemany("UPDATE documents SET conflict=1 WHERE hash=?", rejected)
        corpus.stats["placeholder_test_scaffold"] += len(rejected)
        corpus.db.execute("INSERT INTO completed VALUES(?)", (marker,))
        corpus.report("quality_exclusions_applied")
        commit()
    corpus.db.close()
