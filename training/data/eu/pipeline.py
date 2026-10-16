"""Discover, extract and atomically checkpoint institutional corpora to Hugging Face."""

from __future__ import annotations

import argparse
import copy
import gzip
import importlib
import json
import time
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path

from eu_corpus import (
    EXTRACTION_VERSION,
    Collector,
    clean_text,
    digest,
    normalize_text,
    parse_content,
)
from hf_store import HFStore, json_bytes

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(UTC).isoformat()


def item_key(record):
    """Known source revision plus extraction version; durable across worker restarts."""
    return digest(
        json_bytes(
            {
                key: record.get(key)
                for key in ("source_id", "document_id", "url", "language", "modified_at")
            }
        )
        + EXTRACTION_VERSION.encode()
    )


def round_robin(collections):
    for group in zip_longest(*collections):
        yield from (record for record in group if record is not None)


def pending(record, state, timestamp=None, max_attempts=3):
    key = item_key(record)
    if key in state["completed"]:
        return False
    failure = state["failures"].get(key, {})
    return failure.get("attempts", 0) < max_attempts and failure.get("retry_after_epoch", 0) <= (
        timestamp or time.time()
    )


def extract_record(candidate, data, provenance, source):
    # Source news selectors do not apply to PDF attachments or other document classes.
    selector = source.get("content_selector") if candidate["format"] == "html" else None
    text, title, language, _, kind = parse_content(
        data,
        provenance["content_type"],
        provenance["final_url"],
        source.get("remove_selectors"),
        selector,
    )
    text = clean_text(text)
    if kind not in {"html", "xml", "text", "pdf"} or len(text) < 200:
        raise ValueError(f"No usable full text: kind={kind}, characters={len(text)}")
    text_hash = digest(normalize_text(text).encode())
    # Source revision + extracted content identifies immutable versions.
    identifier = digest((item_key(candidate) + text_hash).encode())
    return {
        "id": identifier,
        "source_id": candidate["source_id"],
        "collection_id": candidate["collection_id"],
        "document_id": candidate["document_id"],
        "url": provenance["final_url"],
        "title": candidate.get("title") or title,
        "language": candidate.get("language") or language or "und",
        "document_type": candidate.get("document_type") or candidate["collection_id"],
        "format": "txt" if kind == "text" else kind,
        "published_at": candidate.get("published_at"),
        "modified_at": candidate.get("modified_at"),
        "retrieved_at": provenance["retrieved_at"],
        "text": text,
        "text_sha256": text_hash,
        "raw_sha256": provenance["raw_sha256"],
        "characters": len(text),
        "words": len(text.split()),
        "extraction_version": EXTRACTION_VERSION,
        "provenance_json": json.dumps(
            {"inventory_record": candidate, "fetch": provenance}, ensure_ascii=False
        ),
    }


def checkpoint(store, state, revision, records, raw_files, extra_files=None, message="Checkpoint"):
    """CAS failure aborts this worker. Never replay stale state over another writer."""
    return store.commit(state, revision, records, raw_files, extra_files, message)


def run_source(
    source_id,
    run_id,
    max_documents=1000,
    work_seconds=3000,
    refresh=False,
    mapping_path=None,
    artifacts_root=None,
    store=None,
):
    mapping = json.loads(Path(mapping_path or ROOT / "pipeline_sources.json").read_text())
    config = mapping["sources"][source_id]
    bounds = mapping["limits"]
    if not 1 <= max_documents <= bounds["max_documents_per_source"]:
        raise ValueError("Document limit exceeds configured authorized bound")
    if not 1 <= work_seconds <= bounds["work_seconds"]:
        raise ValueError("Work duration exceeds configured bound")
    if not run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id
    ):
        raise ValueError("run_id must contain only letters, digits, hyphens and underscores")
    source = next(
        s
        for s in json.loads((ROOT / "sources.json").read_text())["sources"]
        if s["id"] == source_id
    )
    source = copy.deepcopy(source)
    adapter = importlib.import_module(config["adapter"])
    store = store or HFStore(config["repo_id"])
    store.ensure(private=mapping["private"])
    state, revision = store.load()
    output = Path(artifacts_root or ROOT / "artifacts" / "pipeline") / source_id / run_id
    # A container restart can reuse a run_id but never overwrite a previous local attempt.
    output = output / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    collector = Collector(output, max_bytes=bounds["max_inventory_bytes"])
    started = time.monotonic()
    deadline = started + work_seconds
    inventory_cache = {}
    run = {
        "run_id": run_id,
        "source_id": source_id,
        "started_at": now(),
        "status": "running",
        "max_documents": max_documents,
        "work_seconds": work_seconds,
        "attempted": 0,
        "extracted": 0,
        "duplicates": 0,
        "failed": 0,
        "collector_sha256": digest((ROOT / "eu_corpus.py").read_bytes()),
        "pipeline_sha256": digest(Path(__file__).read_bytes()),
        "adapter_sha256": digest(Path(adapter.__file__).read_bytes()),
    }
    state["runs"].append(run)
    state["source_mapping"] = {"source_id": source_id, **config}
    state["limits"] = bounds
    revision = checkpoint(store, state, revision, [], {}, message=f"Start {source_id} run {run_id}")

    # HFStore updates state in place after a successful commit; reacquire nested run references.
    def active_run():
        return state["runs"][-1]

    def fetch_inventory(url):
        if time.monotonic() >= deadline:
            raise TimeoutError("Inventory enumeration reached run deadline")
        if url not in inventory_cache:
            data, provenance = collector.fetch(url, source)
            if data is None:
                raise RuntimeError(f"Inventory fetch failed: {url}: {provenance.get('error')}")
            inventory_cache[url] = data, provenance
        return inventory_cache[url]

    records, raw_files = [], {}
    collection_records = []
    try:
        for collection in config["collections"]:
            if collection in state["collections"] and not refresh:
                saved = store.read_json(
                    state["collections"][collection]["inventory_path"], revision
                )
            else:
                saved = adapter.enumerate_collection(
                    collection, fetch_inventory, limit=bounds["inventory_limit_per_collection"]
                )
                saved["observed_at"] = now()
                inventory_body = json_bytes(saved)
                path = f"inventories/{collection}/{digest(inventory_body)}.json"
                state["collections"][collection] = {
                    "inventory_path": path,
                    "enumeration": saved["enumeration"],
                    "definition": adapter.COLLECTIONS[collection],
                    "updated_at": now(),
                    "candidate_count": len(saved["records"]),
                }
                revision = checkpoint(
                    store,
                    state,
                    revision,
                    [],
                    {},
                    {path: inventory_body},
                    f"Enumerate {source_id}/{collection}",
                )
            collection_records.append(saved["records"])
        inventory_cache.clear()
        collector.max_bytes = bounds["max_document_bytes"]
        since_commit = 0
        for candidate in round_robin(collection_records):
            if active_run()["attempted"] >= max_documents or time.monotonic() >= deadline:
                break
            if not pending(candidate, state, max_attempts=bounds["max_attempts_per_item"]):
                continue
            key = item_key(candidate)
            active_run()["attempted"] += 1
            since_commit += 1
            try:
                data, provenance = collector.fetch(candidate["url"], source)
                if data is None:
                    raise ValueError(provenance.get("error", "fetch_failed"))
                record = extract_record(candidate, data, provenance, source)
                text_hash = record["text_sha256"]
                previous = state["text_index"].get(text_hash)
                if previous:
                    active_run()["duplicates"] += 1
                else:
                    records.append(record)
                    raw_files[f"raw/{record['raw_sha256']}.gz"] = gzip.compress(data, mtime=0)
                    state["text_index"][text_hash] = record["id"]
                    active_run()["extracted"] += 1
                state["completed"][key] = {
                    "status": "duplicate" if previous else "extracted",
                    "record_id": previous or record["id"],
                    "collection_id": candidate["collection_id"],
                    "url": candidate["url"],
                    "document_id": candidate["document_id"],
                    "text_sha256": text_hash,
                    "completed_at": now(),
                }
                state["failures"].pop(key, None)
            except Exception as error:  # noqa: BLE001 - isolate bad documents and persist retry state
                attempts = state["failures"].get(key, {}).get("attempts", 0) + 1
                state["failures"][key] = {
                    "url": candidate["url"],
                    "collection_id": candidate["collection_id"],
                    "attempts": attempts,
                    "error": str(error)[:1000],
                    "failed_at": now(),
                    "retry_after_epoch": time.time() + 300 * attempts,
                }
                active_run()["failed"] += 1
            if since_commit >= bounds["batch_size"]:
                active_run()["heartbeat_at"] = now()
                revision = checkpoint(
                    store,
                    state,
                    revision,
                    records,
                    raw_files,
                    message=f"Checkpoint {source_id}: {active_run()['attempted']} attempts",
                )
                print(
                    json.dumps({"source_id": source_id, "revision": revision, **active_run()}),
                    flush=True,
                )
                records, raw_files, since_commit = [], {}, 0
        active_run()["status"] = "bounded_complete"
        active_run()["finished_at"] = now()
        active_run()["elapsed_seconds"] = round(time.monotonic() - started, 2)
        active_run()["pending_candidates"] = sum(
            item_key(r) not in state["completed"] for group in collection_records for r in group
        )
        revision = checkpoint(
            store,
            state,
            revision,
            records,
            raw_files,
            message=f"Finish bounded {source_id} run {run_id}",
        )
        result = {
            "repo_id": config["repo_id"],
            "revision": revision,
            **active_run(),
            "total_rows": sum(s["rows"] for s in state["shards"]),
            "total_completed": len(state["completed"]),
            "total_failures": len(state["failures"]),
        }
        (output / "result.json").write_bytes(json_bytes(result))
        return result
    finally:
        # On failure do not advance HF state without the associated documents. A stale
        # 'running' marker after termination means resume from the last atomic commit.
        collector.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["ecb", "ema"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument("--work-seconds", type=int, default=3000)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_source(
                args.source, args.run_id, args.max_documents, args.work_seconds, args.refresh
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
