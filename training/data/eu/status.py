"""Read-only Hugging Face progress and pinned-revision Parquet audit.

Run ``uv run status.py both --audit``; add ``--tokens`` for o200k_base counts.
No repositories, progress files, or dataset content are changed by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub.errors import EntryNotFoundError

from eu_corpus import normalize_text
from goal_controller import cached_token_lower_bound
from hf_store import RECORD_SCHEMA, HFStore, json_bytes
from pipeline import load_mapping

ROOT = Path(__file__).resolve().parent


def document_target_status(target, sources):
    """Track saved rows after an immutable baseline; never treat missing reads as zero."""
    if target.get("schema_version") != 1:
        raise ValueError("Incompatible document target")
    baseline = target.get("baseline_total_rows")
    additional = target.get("additional_documents")
    total = target.get("target_total_rows")
    if (
        type(baseline) is not int
        or baseline < 0
        or type(additional) is not int
        or additional < 1
        or type(total) is not int
        or total != baseline + additional
    ):
        raise ValueError("Invalid document target counts")
    expected = {entry["source_id"]: entry for entry in target["sources"]}
    observed = {entry["source_id"]: entry for entry in sources}
    if (
        not expected
        or len(expected) != len(target["sources"])
        or len(observed) != len(sources)
        or set(expected) != set(observed)
    ):
        raise ValueError("Document target requires exactly its baseline sources")
    current = 0
    baseline_sum = 0
    for source, entry in expected.items():
        result = observed[source]
        rows, original = result.get("rows"), entry.get("baseline_rows")
        if (
            result.get("error_type")
            or result.get("repo_id") != entry["repo_id"]
            or not result.get("revision")
            or type(rows) is not int
            or type(original) is not int
            or original < 0
            or rows < original
        ):
            raise ValueError("Missing, incompatible or regressed document source")
        current += rows
        baseline_sum += original
    if baseline_sum != baseline:
        raise ValueError("Document baseline total does not match its sources")
    return {
        "request_id": target["request_id"],
        "baseline_rows": baseline,
        "additional_requested": additional,
        "target_rows": total,
        "current_rows": current,
        "added_rows": current - baseline,
        "remaining_rows": max(0, total - current),
        "status": "reached" if current >= total else "collecting",
        "source_revisions": {key: value["revision"] for key, value in observed.items()},
    }


def audit_shards(store, state, revision, encoder=None):
    """Audit precisely the shards named by progress at its loaded commit."""
    seen = set()
    text_hashes = set()
    counts = Counter()
    languages, formats, collections = Counter(), Counter(), Counter()
    issues = []
    samples = []
    for shard in state["shards"]:
        local = store.api.hf_hub_download(
            repo_id=store.repo_id,
            repo_type="dataset",
            filename=shard["path"],
            revision=revision,
        )
        counts["parquet_bytes"] += Path(local).stat().st_size
        table = pq.read_table(local)
        rows = table.to_pylist()
        if len(rows) != shard["rows"]:
            issues.append({"kind": "shard_row_count", "path": shard["path"]})
        normalized = [{name: row.get(name) for name in RECORD_SCHEMA.names} for row in rows]
        digest = hashlib.sha256(json_bytes(normalized)).hexdigest()
        if digest != shard["sha256"]:
            issues.append({"kind": "shard_content_hash", "path": shard["path"]})
        for row in rows:
            identity = row["id"]
            if identity in seen:
                counts["duplicate_row_ids"] += 1
            seen.add(identity)
            text = row["text"]
            digest = hashlib.sha256(normalize_text(text).encode()).hexdigest()
            if digest != row["text_sha256"]:
                counts["text_hash_mismatches"] += 1
            if digest in text_hashes:
                counts["duplicate_normalized_texts"] += 1
            text_hashes.add(digest)
            characters, words = len(text), len(text.split())
            if characters != row["characters"] or words != row["words"]:
                counts["text_size_mismatches"] += 1
            if row["collection_id"] != shard["collection_id"] or row["format"] != shard["format"]:
                counts["partition_mismatches"] += 1
            counts.update(
                rows=1,
                characters=characters,
                words=words,
                text_utf8_bytes=len(text.encode("utf-8")),
            )
            if encoder is not None:
                counts["tokens"] += len(encoder.encode(text, disallowed_special=()))
            languages[row["language"] or "und"] += 1
            formats[row["format"]] += 1
            collections[row["collection_id"]] += 1
            if len(samples) < 5:
                samples.append(
                    {k: row[k] for k in ("id", "url", "language", "format", "collection_id")}
                )
    missing = {entry["record_id"] for entry in state["completed"].values()} - seen
    bad_index = set(state["text_index"].values()) - seen
    checks = {
        "duplicate_row_ids": counts["duplicate_row_ids"],
        "text_hash_mismatches": counts["text_hash_mismatches"],
        "text_size_mismatches": counts["text_size_mismatches"],
        "partition_mismatches": counts["partition_mismatches"],
        "completed_references_missing": len(missing),
        "text_index_references_missing": len(bad_index),
        "shard_issues": issues,
        "metric_mismatches": {
            key: {"stored": state["metrics"].get(key), "measured": counts[key]}
            for key in ("rows", "characters", "words", "text_utf8_bytes", "parquet_bytes")
            if "metrics" in state and state["metrics"].get(key) != counts[key]
        },
    }
    return {
        "revision": revision,
        "passed": not any(checks.values()),
        "shards_checked": len(state["shards"]),
        "rows": counts["rows"],
        "unique_row_ids": len(seen),
        "characters": counts["characters"],
        "words": counts["words"],
        "text_utf8_bytes": counts["text_utf8_bytes"],
        "parquet_bytes": counts["parquet_bytes"],
        "duplicate_normalized_texts": counts["duplicate_normalized_texts"],
        "languages": dict(languages),
        "formats": dict(formats),
        "collections": dict(collections),
        "checks": checks,
        "samples": samples,
        **({"tokenizer": "o200k_base", "tokens": counts["tokens"]} if encoder else {}),
    }


def cached_token_status(state, repo_id, source_id, control_store, control_revision):
    """Read one pinned cache and report only entries matching current shards."""
    try:
        cache = json.loads(
            control_store._download_bytes(f"tokens/{source_id}.json", control_revision)
        )
    except EntryNotFoundError:
        cache = None
    tokens = cached_token_lower_bound(state, cache, repo_id)
    matched = [
        shard
        for shard in state["shards"]
        if cache is not None
        and (entry := cache["shards"].get(shard["sha256"])) is not None
        and entry.get("path") == shard["path"]
        and entry.get("rows") == shard["rows"]
    ]
    return {
        "encoding": "o200k_base",
        "control_revision": control_revision,
        "exact_tokens_lower_bound": tokens,
        "matched_rows": sum(shard["rows"] for shard in matched),
        "matched_shards": len(matched),
        "total_shards": len(state["shards"]),
        "complete_for_source_revision": cache is not None and len(matched) == len(state["shards"]),
        "cache_present": cache is not None,
        "audit_summary": cache.get("summary") if cache is not None else None,
    }


def source_status(
    source_id, config, audit=False, encoder=None, control_store=None, control_revision=None
):
    store = HFStore(config["repo_id"])
    state, revision = store.load()
    info = store.api.repo_info(repo_id=store.repo_id, repo_type="dataset", revision=revision)
    rows = sum(shard["rows"] for shard in state["shards"])
    collections = {}
    for name in config["collections"]:
        collection = state["collections"].get(name, {})
        enumeration = collection.get("enumeration", {})
        collections[name] = {
            "candidates": collection.get("candidate_count"),
            "reported_total": enumeration.get("reported_total"),
            "enumeration_complete": enumeration.get("complete"),
            "updated_at": collection.get("updated_at"),
            "completed": sum(x["collection_id"] == name for x in state["completed"].values()),
            "failures": sum(x["collection_id"] == name for x in state["failures"].values()),
            "rows": sum(x["rows"] for x in state["shards"] if x["collection_id"] == name),
        }
    result = {
        "source_id": source_id,
        "repo_id": store.repo_id,
        "url": f"https://huggingface.co/datasets/{store.repo_id}",
        "revision": revision,
        "visibility": "private" if info.private else "public",
        "rows": rows,
        "metrics": state.get("metrics"),
        "collection_metrics": state.get("collection_metrics"),
        "metrics_history_count": len(state.get("metrics_history", [])),
        "recent_metrics": state.get("metrics_history", [])[-5:],
        "campaigns": state.get("campaigns", {}),
        "shards": len(state["shards"]),
        "completed": len(state["completed"]),
        "duplicates": sum(x.get("status") == "duplicate" for x in state["completed"].values()),
        "failures": len(state["failures"]),
        "download_queue": {
            "stored_raw_documents": len(state.get("download_queue", {})),
            "awaiting_first_parse": sum(
                key not in state["failures"] for key in state.get("download_queue", {})
            ),
            "retained_after_parse_failure": sum(
                key in state["failures"] for key in state.get("download_queue", {})
            ),
        },
        "collections": collections,
        "run_count": len(state["runs"]),
        "recent_runs": state["runs"][-5:],
        "failure_samples": [
            {k: value.get(k) for k in ("url", "collection_id", "attempts", "failed_at")}
            for value in list(state["failures"].values())[:5]
        ],
    }
    if audit or encoder is not None:
        result["audit"] = audit_shards(store, state, revision, encoder)
    if control_store is not None:
        if not control_revision:
            raise ValueError("Pinned control revision required")
        result["cached_tokens"] = cached_token_status(
            state, store.repo_id, source_id, control_store, control_revision
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        choices=["both", "all", "ecb", "ema", "eurlex", "commission-press"],
        nargs="?",
        default="both",
    )
    parser.add_argument("--mapping", type=Path, default=ROOT / "pipeline_sources.json")
    parser.add_argument(
        "--document-target", type=Path, help="Track additional saved rows against a baseline JSON"
    )
    parser.add_argument(
        "--audit", action="store_true", help="Download and verify all stored shards"
    )
    parser.add_argument(
        "--tokens", action="store_true", help="Implies --audit; count o200k_base tokens"
    )
    parser.add_argument(
        "--cached-tokens",
        action="store_true",
        help="Read matching exact token caches without recounting",
    )
    args = parser.parse_args()
    mapping = load_mapping(args.mapping)
    encoder = None
    if args.tokens:
        import tiktoken

        encoder = tiktoken.get_encoding("o200k_base")
    sources = (
        list(mapping["sources"])
        if args.source == "all"
        else (["ecb", "ema"] if args.source == "both" else [args.source])
    )
    output = {"observed_at": datetime.now(UTC).isoformat(), "sources": []}
    failed = False
    control_store, control_revision = None, None
    if args.cached_tokens:
        config = json.loads((ROOT / "collection_goal.json").read_text())
        control_store = HFStore(config["control_repo"])
        try:
            control, control_revision = control_store.load()
            schedule = json.loads(
                control_store._download_bytes("tokens/schedule.json", control_revision)
            )
            goal = control.get("collection_goal", {})
            output["control"] = {
                "revision": control_revision,
                "goal_status": goal.get("status"),
                "campaigns_reserved": len(goal.get("jobs", [])),
                "reserved_documents": goal.get("reserved_documents"),
                "audit_schedule": schedule.get("summary"),
            }
        except Exception as error:  # noqa: BLE001 - do not substitute zero for failed reads
            output["control"] = {"error_type": type(error).__name__}
            failed = True
            control_store = None
    for source in sources:
        try:
            result = source_status(
                source,
                mapping["sources"][source],
                args.audit,
                encoder,
                control_store,
                control_revision,
            )
            failed |= not result.get("audit", {}).get("passed", True)
        except Exception as error:  # noqa: BLE001 - produce usable JSON even if one repo is unavailable
            # Do not expose arbitrary HTTP exception bodies or authentication details.
            result = {"source_id": source, "error_type": type(error).__name__}
            failed = True
        output["sources"].append(result)
    available_metrics = [source.get("metrics") for source in output["sources"]]
    if all(available_metrics):
        output["combined_metrics"] = {
            key: sum(metrics[key] for metrics in available_metrics) for key in available_metrics[0]
        }
    if args.cached_tokens and all("cached_tokens" in source for source in output["sources"]):
        output["exact_tokens_lower_bound"] = sum(
            source["cached_tokens"]["exact_tokens_lower_bound"] for source in output["sources"]
        )
    if args.document_target:
        try:
            output["document_target"] = document_target_status(
                json.loads(args.document_target.read_text()), output["sources"]
            )
        except Exception as error:  # noqa: BLE001 - partial reads must not claim target completion
            output["document_target"] = {
                "status": "unavailable",
                "error_type": type(error).__name__,
            }
            failed = True
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
