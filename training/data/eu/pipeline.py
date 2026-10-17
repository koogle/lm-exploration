"""Discover, extract and atomically checkpoint institutional corpora to Hugging Face."""

from __future__ import annotations

import argparse
import copy
import gzip
import importlib
import json
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from itertools import islice
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


def round_robin(collections, weights=None):
    """Weighted fair iteration; every nonempty collection gets at least one slot."""
    weights = list(weights) if weights is not None else [1] * len(collections)
    if len(weights) != len(collections) or any(
        not isinstance(w, int) or isinstance(w, bool) or not 1 <= w <= 20 for w in weights
    ):
        raise ValueError("Collection weights must be integers from 1 to 20")
    active = [(iter(group), weight) for group, weight in zip(collections, weights, strict=True)]
    while active:
        remaining = []
        for iterator, weight in active:
            group = list(islice(iterator, weight))
            yield from group
            if len(group) == weight:
                remaining.append((iterator, weight))
        active = remaining


def pending(record, state, timestamp=None, max_attempts=3, adapter=None):
    key = item_key(record)
    if key in state["completed"]:
        return False
    failure = state["failures"].get(key, {})
    if failure.get("retry_after_epoch", 0) > (timestamp or time.time()):
        return False
    if failure.get("attempts", 0) < max_attempts:
        return True
    # Newly verified official replacements get one opportunity even after the
    # original inventory URL exhausted its retry budget. Never reset old attempts.
    alternatives = getattr(adapter, "alternate_document_urls", lambda _: [])(record)
    return failure.get("error") == "http_404" and bool(
        set(alternatives) - set(failure.get("attempted_alternates", []))
    )


def rate_limit_retry_epoch(provenance, timestamp):
    """Honor Retry-After seconds/date; use a 15-minute fallback when absent/invalid."""
    headers = provenance.get("headers") or provenance.get("response_headers") or {}
    value = provenance.get("retry_after") or next(
        (v for k, v in headers.items() if k.lower() == "retry-after"), None
    )
    if value is not None:
        try:
            return timestamp + max(0, int(value))
        except (TypeError, ValueError):
            try:
                date = parsedate_to_datetime(str(value))
                if date.tzinfo is None:
                    date = date.replace(tzinfo=UTC)
                return max(timestamp, date.timestamp())
            except (TypeError, ValueError, OverflowError):
                pass
    return timestamp + 900


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


def run_budget(previous, max_documents, work_seconds, batch_size, timestamp):
    """Bound a logical run across preemptions, including possibly lost batches."""
    if not previous:
        return max_documents, work_seconds, 0, 0
    first = previous[0]
    document_bound = min(max_documents, first["max_documents"])
    seconds_bound = min(work_seconds, first["work_seconds"])
    elapsed = max(0, timestamp - datetime.fromisoformat(first["started_at"]).timestamp())
    persisted = sum(run["attempted"] for run in previous)
    # A running marker can hide up to one fetched but not checkpointed batch.
    reserved = sum(
        run.get("checkpoint_batch_size", batch_size)
        for run in previous
        if run["status"] == "running"
    )
    return (
        max(0, document_bound - persisted - reserved),
        max(0, seconds_bound - elapsed),
        persisted,
        reserved,
    )


def run_result(config, state, revision, run):
    return {
        "repo_id": config["repo_id"],
        "revision": revision,
        **run,
        "metrics": state.get("metrics"),
        "collection_metrics": state.get("collection_metrics"),
        "total_rows": sum(s["rows"] for s in state["shards"]),
        "total_completed": len(state["completed"]),
        "total_failures": len(state["failures"]),
    }


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
    previous = [run for run in state["runs"] if run["run_id"] == run_id]
    terminal = next(
        (run for run in reversed(previous) if run["status"] == "bounded_complete"), None
    )
    if terminal is not None:
        return run_result(config, state, revision, terminal)
    budget_checked_at = time.monotonic()
    remaining_documents, remaining_seconds, persisted_attempts, reserved_attempts = run_budget(
        previous, max_documents, work_seconds, bounds["batch_size"], time.time()
    )
    if previous and (remaining_documents == 0 or remaining_seconds == 0):
        # Finalize the last durable attempt without opening a source HTTP client.
        last = previous[-1]
        last["status"] = "bounded_complete"
        last["finished_at"] = now()
        last["stop_reason"] = "logical_run_budget_exhausted"
        last["logical_run_attempts_persisted"] = persisted_attempts
        last["logical_run_attempts_reserved"] = reserved_attempts
        revision = checkpoint(
            store, state, revision, [], {}, message=f"Finish exhausted {source_id} run {run_id}"
        )
        return run_result(config, state, revision, last)
    if state.get("source_retry_after_epoch", 0) > time.time():
        raise ValueError(
            f"Source {source_id} is rate-limited until epoch "
            f"{state['source_retry_after_epoch']}; no collection attempted"
        )
    output = Path(artifacts_root or ROOT / "artifacts" / "pipeline") / source_id / run_id
    # A container restart can reuse a run_id but never overwrite a previous local attempt.
    output = output / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    collector = Collector(output, max_bytes=bounds["max_inventory_bytes"])
    started = time.monotonic()
    deadline = budget_checked_at + remaining_seconds
    inventory_cache = {}
    run = {
        "run_id": run_id,
        "source_id": source_id,
        "source_config": copy.deepcopy(source),
        "started_at": now(),
        "status": "running",
        "max_documents": max_documents,
        "work_seconds": work_seconds,
        "checkpoint_batch_size": bounds["batch_size"],
        "logical_run_started_at": previous[0]["started_at"] if previous else now(),
        "logical_run_attempts_persisted": persisted_attempts,
        "logical_run_attempts_reserved": reserved_attempts,
        "remaining_documents_at_start": remaining_documents,
        "remaining_seconds_at_start": remaining_seconds,
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
                    collection,
                    fetch_inventory,
                    limit=min(
                        config.get(
                            "inventory_limit_per_collection",
                            bounds["inventory_limit_per_collection"],
                        ),
                        bounds["inventory_limit_per_collection"],
                    ),
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
        # Filter before scheduling so completed prefixes cannot consume weight slots.
        supports = getattr(adapter, "supported_record", lambda _: True)
        active_run()["unsupported_candidates"] = sum(
            not supports(record) for group in collection_records for record in group
        )
        eligible = [
            [
                record
                for record in group
                if supports(record)
                and pending(
                    record, state, max_attempts=bounds["max_attempts_per_item"], adapter=adapter
                )
            ]
            for group in collection_records
        ]
        weights = [
            config.get("collection_weights", {}).get(name, 1) for name in config["collections"]
        ]
        active_run()["collection_weights"] = dict(zip(config["collections"], weights, strict=True))
        for candidate in round_robin(eligible, weights):
            if active_run()["attempted"] >= remaining_documents or time.monotonic() >= deadline:
                break
            if not pending(
                candidate, state, max_attempts=bounds["max_attempts_per_item"], adapter=adapter
            ):
                continue
            key = item_key(candidate)
            active_run()["attempted"] += 1
            since_commit += 1
            provenance = {}
            attempted_alternates = list(
                state["failures"].get(key, {}).get("attempted_alternates", [])
            )
            rate_limited = False
            try:
                data, provenance = collector.fetch(candidate["url"], source)
                if data is None and (
                    str(provenance.get("http_status", provenance.get("status"))) == "404"
                    or provenance.get("error") == "http_404"
                ):
                    alternatives = getattr(adapter, "alternate_document_urls", lambda _: [])(
                        candidate
                    )
                    if alternatives:
                        original_fetch = {"url": candidate["url"], **provenance}
                        attempted_alternates.append(alternatives[0])
                        data, provenance = collector.fetch(alternatives[0], source)
                        provenance["original_fetch"] = original_fetch
                        if data is not None:
                            validator = getattr(adapter, "validate_alternate", None)
                            if validator:
                                validator(candidate, data)
                rate_limited = (
                    str(provenance.get("http_status", provenance.get("status"))) == "429"
                    or provenance.get("error") == "http_429"
                )
                if rate_limited:
                    raise ValueError("http_429")
                if data is None:
                    raise ValueError(provenance.get("error", "fetch_failed"))
                parse_data, parse_provenance = data, provenance
                transform = getattr(adapter, "extract_document", None)
                if transform:
                    parse_data, transform_metadata = transform(candidate, data)
                    parse_provenance = {
                        **provenance,
                        "content_type": "text/html",
                        "original_content_type": provenance.get("content_type"),
                        "adapter_transform": transform_metadata,
                    }
                record = extract_record(candidate, parse_data, parse_provenance, source)
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
                    "attempted_alternates": sorted(set(attempted_alternates)),
                    "error": str(error)[:1000],
                    "failed_at": now(),
                    "retry_after_epoch": (
                        rate_limit_retry_epoch(provenance, time.time())
                        if rate_limited
                        else time.time() + 300 * attempts
                    ),
                }
                active_run()["failed"] += 1
                if rate_limited:
                    state["source_retry_after_epoch"] = state["failures"][key]["retry_after_epoch"]
                    active_run()["status"] = "paused_rate_limit"
                    active_run()["retry_after_epoch"] = state["failures"][key]["retry_after_epoch"]
                    # Flush accumulated rows and this failure together below; never
                    # request the next candidate after a server rate-limit response.
                    break
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
        if active_run()["status"] == "running":
            active_run()["status"] = "bounded_complete"
        active_run()["finished_at"] = now()
        active_run()["elapsed_seconds"] = round(time.monotonic() - started, 2)
        active_run()["pending_candidates"] = sum(
            item_key(r) not in state["completed"] for group in collection_records for r in group
        )
        active_run()["remaining_supported_candidates"] = sum(
            supports(r)
            and pending(
                r,
                state,
                timestamp=float("inf"),
                max_attempts=bounds["max_attempts_per_item"],
                adapter=adapter,
            )
            for group in collection_records
            for r in group
        )
        revision = checkpoint(
            store,
            state,
            revision,
            records,
            raw_files,
            message=f"Finish bounded {source_id} run {run_id}",
        )
        result = run_result(config, state, revision, active_run())
        (output / "result.json").write_bytes(json_bytes(result))
        return result
    finally:
        # On failure do not advance HF state without the associated documents. A stale
        # 'running' marker after termination means resume from the last atomic commit.
        collector.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["ecb", "ema", "eurlex", "commission-press"])
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
