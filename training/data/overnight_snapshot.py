"""Fail-closed data gate and immutable eligible-only overnight snapshot."""

import gzip
import json
import sqlite3
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from .overnight_corpus import SCHEMA
from .subword import TOKENIZER_SHA256, digest

CODE_REQUIRED = 6_120_008_192  # 6B code targets, 2% margin and one context window.


def weights_from(manifest):
    previous = json.loads(Path(manifest).read_text())["sources"]
    weights = {
        ("code" if name == "code_python" else name): (
            Fraction(60)
            if name == "code_python"
            else Fraction(value.get("weight_percent_exact", str(value["weight_percent"])))
            * Fraction(7, 8)
        )
        for name, value in previous.items()
    }
    weights["eu"] = Fraction(5)
    if sum(weights.values()) != 100 or weights["code"] != 60:
        raise ValueError("Expected prior 60/40 mix")
    return weights


def check_variety(audit_db):
    rows = audit_db.execute(
        "SELECT stratum,SUM(tokens) FROM eligible WHERE fresh=1 AND source='code' AND split='train' GROUP BY stratum"
    ).fetchall()
    languages, tasks = defaultdict(int), defaultdict(int)
    for stratum, tokens in rows:
        _, language, task = stratum.split("/", 2)
        languages[language] += tokens
        tasks[task] += tokens
    # These are declared operational thresholds, not proof of model capabilities.
    if sum(languages.values()) < CODE_REQUIRED:
        raise ValueError(
            f"Need >=6.12B new eligible code targets including margin; got {sum(languages.values())}"
        )
    if sum(n >= 10_000_000 for language, n in languages.items() if language != "diff") < 8:
        raise ValueError("Need eight languages with at least 10M eligible targets each")
    if sum(n >= 10_000_000 for n in tasks.values()) < 3 or tasks.get("tests", 0) < 50_000_000:
        raise ValueError("Need three path-task families >=10M each and >=50M test-code targets")
    if sum(min(n, 2_400_000_000) for n in languages.values()) < CODE_REQUIRED:
        raise ValueError("Cannot retain 6.12B eligible targets with a 2.4B/language cap")
    return {"languages": dict(languages), "path_tasks": dict(tasks)}


def freeze(audit_folder, destination, old_manifest, database_paths):
    from trainer.continued_pretraining import token_targets

    audit_folder, destination = Path(audit_folder), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise ValueError("Refuse to overwrite frozen snapshot")
    report = json.loads((audit_folder / "report.json").read_text())
    if (
        report["stage"] != "audited_not_training_ready"
        or report["tokenizer_sha256"] != TOKENIZER_SHA256
    ):
        raise ValueError("Missing completed audit or tokenizer mismatch")
    if not report["benchmarks"] or len(report["benchmarks"]) != 2:
        raise ValueError("Incomplete HumanEval/MBPP screen")
    audit_db = sqlite3.connect(f"file:{audit_folder / 'audit.sqlite'}?mode=ro", uri=True)
    variety = check_variety(audit_db)
    # Tokenizer compatibility is frozen; reject obvious failures/poor coverage.
    diagnostic = {s: v for s, v in report["tokenizer"].items() if s.startswith("code/")}
    if not diagnostic or any(v["unknowns"] for v in diagnostic.values()):
        raise ValueError("Code tokenizer diagnostic missing or contains UNK")
    total_chars = sum(v["characters"] for v in diagnostic.values())
    if not total_chars or sum(v["tokens"] for v in diagnostic.values()) / total_chars > 1:
        raise ValueError("Code tokenizer fragmentation exceeds one token/character")
    covered = {
        s.split("/")[1]
        for s, v in report["heldout"].items()
        if s.startswith("code/") and v["parents"] >= 4
    }
    if (
        sum(
            language in covered
            for language, n in variety["languages"].items()
            if n >= 10_000_000 and language != "diff"
        )
        < 8
    ):
        raise ValueError("Eight material code languages need >=4 held-out parents each")
    weights = weights_from(old_manifest)
    quotas = token_targets(weights, 10_000_000_000)
    final_path = destination / "raw.sqlite"
    if final_path.exists():
        raise ValueError("Unpublished final database exists; inspect before retry")
    path = destination / "raw.building.sqlite"
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA cache_size=-1048576")
    db.executescript(SCHEMA)
    handles = {
        name: sqlite3.connect(f"file:{p}?mode=ro", uri=True) for name, p in database_paths.items()
    }
    for handle in handles.values():
        handle.execute("PRAGMA cache_size=-1048576")
    totals, languages, task_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    with db:
        for origin, key, source, split, tokens, stratum, fresh in audit_db.execute(
            "SELECT origin,hash,source,split,tokens,stratum,fresh FROM eligible ORDER BY fresh DESC,hash"
        ):
            name = "code" if source == "code_python" else source
            if name not in weights or (name == "code" and not fresh):
                continue
            if split == "train":
                cap = 1 if name == "code" else 2
                needed = (quotas[name] * 102 + 100 * cap - 1) // (100 * cap) + 8192
                if totals[name] >= needed:
                    continue
                if name == "code":
                    _, language, task = stratum.split("/", 2)
                    if languages[language] + tokens > 2_400_000_000:
                        continue
                    languages[language] += tokens
                    task_counts[task] += tokens
            row = (
                handles[origin]
                .execute(
                    "SELECT hash,source,split,parent,provenance,text,loss_tokens,conflict FROM documents WHERE hash=?",
                    (key,),
                )
                .fetchone()
            )
            if not row or row[2] != split or row[6] != tokens or row[7]:
                raise ValueError("Audit/source row mismatch")
            row = list(row)
            row[1] = name
            db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)", row)
            if split == "train":
                totals[name] += tokens
    # Holdout is frozen from the same eligible records. Use 16 parents/stratum
    # for frequent evaluation; retain the larger audit set for final evaluation.
    windows = 0
    parents = defaultdict(set)
    counts = defaultdict(int)
    with (
        gzip.open(audit_folder / "heldout-windows.jsonl.gz", "rt") as source,
        gzip.open(destination / "heldout.jsonl.gz", "wt") as out,
    ):
        for line in source:
            row = json.loads(line)
            stratum = row["stratum"]
            name = stratum.split("/")[0]
            if name not in weights:
                continue
            parent = row["parent"]
            if parent not in parents[stratum] and len(parents[stratum]) >= 16:
                continue
            # Old code validation was excluded from training but is still useful
            # to measure old Python capability; new language strata are separate.
            if name == "code" and not row["stratum"].startswith("code/"):
                raise ValueError("Invalid code stratum")
            parents[stratum].add(parent)
            row["source"] = name
            out.write(json.dumps(row) + "\n")
            counts[name] += len(row["tokens"]) - 1
            windows += 1
    if set(counts) != set(weights):
        raise ValueError(f"Heldout source coverage missing: {set(weights) - set(counts)}")
    summary = {}
    for name, weight in weights.items():
        split_counts = dict(
            db.execute(
                "SELECT split,SUM(loss_tokens) FROM documents WHERE source=? GROUP BY split",
                (name,),
            )
        )
        cap = 1 if name == "code" else 2
        if split_counts.get("train", 0) * cap < quotas[name] * 1.02:
            raise ValueError(f"Insufficient post-filter capacity for {name}: {split_counts}")
        if split_counts.get("validation", 0) <= 0:
            raise ValueError("Missing held-out documents: " + name)
        summary[name] = {
            "weight_percent": float(weight),
            "weight_percent_exact": str(weight),
            "max_passes": cap,
            "splits": {s: {"loss_tokens": split_counts.get(s, 0)} for s in ["train", "validation"]},
        }
    if (
        sum(n >= 10_000_000 for language, n in languages.items() if language != "diff") < 8
        or task_counts.get("tests", 0) < 50_000_000
    ):
        raise ValueError("Selected snapshot lost required code variety")
    db.close()
    for handle in handles.values():
        handle.close()
    audit_db.close()
    path.replace(final_path)
    path = final_path
    manifest = {
        "format": "scholarly-raw-v1",
        "training_ready": False,
        "max_passes": 2,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "database": {"path": path.name, "sha256": digest(path)},
        "sources": summary,
        "heldout": {
            "path": "heldout.jsonl.gz",
            "sha256": digest(destination / "heldout.jsonl.gz"),
            "tokenizer_sha256": TOKENIZER_SHA256,
            "windows": windows,
            "parents_per_stratum": 16,
        },
        "phase_target_tokens": 10_000_000_000,
        "readiness_gate_passed": True,
        "audit_report_sha256": digest(audit_folder / "report.json"),
        "code_languages": dict(languages),
        "code_path_tasks": dict(task_counts),
        "variety_available": variety,
        "patch_policy": {
            "representation": "git_patch",
            "related_version_overlap_allowed": True,
            "exact_normalized_document_dedup": True,
            "selected_train_tokens": languages.get("diff", 0),
            "counts_as_programming_language": False,
        },
        "pending": [
            "Lexical HumanEval/MBPP screening does not prove semantic contamination absent",
            "Whitespace dedup does not remove all near/semantic clones",
            "Non-code family identity is inherited; edition/URL aliases may remain",
            "Path-based task variety is a proxy; this is not instruction tuning",
            "Non-code may reuse earlier corpus; two passes applies to this phase, not lifetime",
            "Patch records may overlap code versions; eligible document tokens are not unique code-span tokens",
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    table = "| Source | Mix | Eligible train tokens (patch versions may overlap) | Pass cap | Requested targets |\n|---|---:|---:|---:|---:|\n"
    for name, value in summary.items():
        table += f"| {name} | {value['weight_percent']:.4f}% | {value['splits']['train']['loss_tokens']:,} | {value['max_passes']} | {quotas[name]:,} |\n"
    (destination / "capacity.md").write_text("# Frozen usable-token capacity\n\n" + table)
    return manifest
