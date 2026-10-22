"""One paced downloader thread, durable raw staging, and a separate parser.

Only the calling thread writes HF state. The downloader handles no mutable
progress and spools at most one checkpoint batch locally while parsing proceeds.
"""

from __future__ import annotations

import copy
import gzip
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pipeline
from eu_corpus import digest
from hf_store import resolve_raw_blob_path
from pipeline import checkpoint, item_key, now, pending, rate_limit_retry_epoch


def limited(provenance):
    return (
        str(provenance.get("http_status", provenance.get("status"))) == "429"
        or provenance.get("error") == "http_429"
    )


def download_batch(batch, collector, source, adapter, spool, deadline, stopped):
    """No parsing, HF writes, or progress mutations on the network thread."""
    outcomes = []
    for candidate, attempted_alternates in batch:
        if stopped.is_set() or time.monotonic() >= deadline:
            break
        outcome = {
            "candidate": candidate,
            "attempted_alternates": attempted_alternates,
            "provenance": {},
        }
        try:
            data, provenance = collector.fetch(candidate["url"], source)
            outcome["provenance"] = provenance
            if data is None and (
                str(provenance.get("http_status", provenance.get("status"))) == "404"
                or provenance.get("error") == "http_404"
            ):
                alternatives = getattr(adapter, "alternate_document_urls", lambda _: [])(candidate)
                if alternatives:
                    original = {"url": candidate["url"], **provenance}
                    attempted_alternates.append(alternatives[0])
                    data, provenance = collector.fetch(alternatives[0], source)
                    provenance["original_fetch"] = original
                    outcome["provenance"] = provenance
                    outcome["used_alternate"] = True
            if limited(provenance):
                raise ValueError("http_429")
            if data is None:
                raise ValueError(provenance.get("error", "fetch_failed"))
            if len(data) > collector.max_bytes or digest(data) != provenance["raw_sha256"]:
                raise ValueError("Downloaded raw body size or digest mismatch")
            path = spool / item_key(candidate)
            path.write_bytes(data)
            outcome["local_path"] = str(path)
        except Exception as error:  # noqa: BLE001 - failure is durably recorded by writer
            outcome["error"] = str(error)[:1000]
        outcomes.append(outcome)
        if limited(outcome["provenance"]):
            break  # Never fetch another URL after a source-wide rate limit.
    return outcomes


def record_failure(state, outcome, error):
    candidate, provenance = outcome["candidate"], outcome["provenance"]
    key = item_key(candidate)
    attempts = state["failures"].get(key, {}).get("attempts", 0) + 1
    state["failures"][key] = {
        "url": candidate["url"],
        "collection_id": candidate["collection_id"],
        "attempts": attempts,
        "attempted_alternates": sorted(set(outcome["attempted_alternates"])),
        "error": str(error)[:1000],
        "failed_at": now(),
        "retry_after_epoch": (
            rate_limit_retry_epoch(provenance, time.time())
            if limited(provenance)
            else time.time() + 300 * attempts
        ),
    }
    run = state["runs"][-1]
    run["failed"] += 1
    if limited(provenance):
        retry = state["failures"][key]["retry_after_epoch"]
        state["source_retry_after_epoch"] = retry
        run.update(status="paused_rate_limit", retry_after_epoch=retry)


def exhausted(state, key, max_attempts):
    return state["failures"].get(key, {}).get("attempts", 0) >= max_attempts


def parse_staged(
    store, state, revision, keys, source, adapter, deadline, bounds, local_paths=None, replay=False
):
    """Parse only raw bodies already committed with queue metadata."""
    records, processed = [], 0
    for key in keys:
        if time.monotonic() >= deadline:
            break  # Durable queued raw remains available to the next worker.
        processed += 1
        run = state["runs"][-1]
        if replay:
            run["attempted"] += 1
            run["queue_replayed"] = run.get("queue_replayed", 0) + 1
        entry = state["download_queue"][key]
        candidate, provenance = entry["candidate"], entry["provenance"]
        try:
            body = (
                local_paths[key].read_bytes()
                if local_paths and key in local_paths
                else gzip.decompress(store._download_bytes(entry["raw_path"], revision))
            )
            if len(body) > bounds["max_document_bytes"] or digest(body) != provenance["raw_sha256"]:
                raise ValueError("Durable queued raw body size or digest mismatch")
        except Exception as error:  # noqa: BLE001 - one bad blob must not wedge every replay
            # Drop the unreadable staged body so a later worker downloads it again.
            record_failure(state, entry, f"staged_raw_unreadable: {error}")
            del state["download_queue"][key]
            continue
        try:
            if entry.get("used_alternate"):
                validator = getattr(adapter, "validate_alternate", None)
                if validator:
                    validator(candidate, body)
            data, parse_provenance = body, provenance
            transform = getattr(adapter, "extract_document", None)
            if transform:
                data, metadata = transform(candidate, body)
                parse_provenance = {
                    **provenance,
                    "content_type": "text/html",
                    "original_content_type": provenance.get("content_type"),
                    "adapter_transform": metadata,
                }
            record = pipeline.extract_record(candidate, data, parse_provenance, source)
            text_hash = record["text_sha256"]
            previous = state["text_index"].get(text_hash)
            if previous:
                run["duplicates"] += 1
            else:
                records.append(record)
                state["text_index"][text_hash] = record["id"]
                run["extracted"] += 1
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
            del state["download_queue"][key]
        except Exception as error:  # noqa: BLE001 - retain raw for bounded parse retry
            record_failure(state, entry, error)
            if exhausted(state, key, bounds["max_attempts_per_item"]):
                # The failure record keeps raw_path; the queue holds only retryable work.
                state["failures"][key]["raw_path"] = entry["raw_path"]
                del state["download_queue"][key]
    if not processed:
        return revision
    state["runs"][-1]["heartbeat_at"] = now()
    return checkpoint(
        store, state, revision, records, {}, message="Checkpoint parsed document queue"
    )


def run_queued_documents(
    *,
    store,
    state,
    revision,
    collector,
    source,
    adapter,
    candidates,
    config,
    bounds,
    maximum,
    deadline,
    output,
):
    run = state["runs"][-1]
    run["document_pipeline"] = "durable_download_queue_v1"
    run["document_queue_sha256"] = digest(Path(__file__).read_bytes())
    queue = state.setdefault("download_queue", {})
    # A lowered attempt bound can exhaust retained entries; keep only retryable work queued.
    for key in [k for k in queue if exhausted(state, k, bounds["max_attempts_per_item"])]:
        state["failures"][key]["raw_path"] = queue.pop(key)["raw_path"]
    # Queue entries are immutable source revisions, not just URLs. Failed parses
    # keep the body and the existing retry budget/cooldown; completed items vanish
    # only in the same atomic commit that stores their parsed result.
    replay = [
        key
        for key, entry in queue.items()
        if entry["candidate"]["collection_id"] in config["collections"]
        and pending(
            entry["candidate"], state, max_attempts=bounds["max_attempts_per_item"], adapter=adapter
        )
    ][:maximum]
    # Reserve budget for replay before allocating any new network work.
    allocated = len(replay)
    iterator = iter(candidates)
    seen = set(queue) | set(state["completed"])
    batch_size = bounds["batch_size"]
    stopped = Event()
    spool = Path(output) / "download-spool"
    spool.mkdir()

    def next_batch():
        nonlocal allocated
        batch = []
        if state.get("source_retry_after_epoch", 0) > time.time():
            return batch
        for candidate in iterator:
            if allocated >= maximum or time.monotonic() >= deadline:
                break
            key = item_key(candidate)
            if key in seen or not pending(
                candidate, state, max_attempts=bounds["max_attempts_per_item"], adapter=adapter
            ):
                continue
            seen.add(key)
            allocated += 1
            batch.append(
                (
                    copy.deepcopy(candidate),
                    list(state["failures"].get(key, {}).get("attempted_alternates", [])),
                )
            )
            if len(batch) == batch_size:
                break
        return batch

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="source-download") as downloader:
        future = None

        def submit_next():
            batch = next_batch()
            return (
                downloader.submit(
                    download_batch, batch, collector, source, adapter, spool, deadline, stopped
                )
                if batch
                else None
            )

        try:
            for start in range(0, len(replay), batch_size):
                if time.monotonic() >= deadline:
                    break
                revision = parse_staged(
                    store,
                    state,
                    revision,
                    replay[start : start + batch_size],
                    source,
                    adapter,
                    deadline,
                    bounds,
                    replay=True,
                )
            future = submit_next()
            while future is not None:
                outcomes = future.result()
                keys, raw_files, local_paths = [], {}, {}
                for outcome in outcomes:
                    state["runs"][-1]["attempted"] += 1
                    if "error" in outcome:
                        record_failure(state, outcome, outcome["error"])
                        continue
                    path = Path(outcome.pop("local_path"))
                    body = path.read_bytes()
                    raw_path = resolve_raw_blob_path(state, outcome["provenance"]["raw_sha256"])
                    raw_files[raw_path] = gzip.compress(body, mtime=0)
                    key = item_key(outcome["candidate"])
                    state["download_queue"][key] = {
                        **outcome,
                        "raw_path": raw_path,
                        "downloaded_at": now(),
                    }
                    keys.append(key)
                    local_paths[key] = path
                if outcomes:
                    state["runs"][-1]["heartbeat_at"] = now()
                    state["runs"][-1]["downloaded"] = state["runs"][-1].get("downloaded", 0) + len(
                        keys
                    )
                    revision = checkpoint(
                        store,
                        state,
                        revision,
                        [],
                        raw_files,
                        message="Stage downloaded document queue",
                    )
                raw_files.clear()
                # The producer overlaps source I/O with parsing; it never writes
                # HF. At most one batch is speculative, covered by run_budget's
                # existing checkpoint-batch reservation after a preemption.
                future = submit_next() if outcomes else None
                if keys:
                    revision = parse_staged(
                        store,
                        state,
                        revision,
                        keys,
                        source,
                        adapter,
                        deadline,
                        bounds,
                        local_paths,
                    )
                for path in local_paths.values():
                    path.unlink()
        finally:
            stopped.set()
            if future is not None:
                future.cancel()
            # Executor joins before the caller closes the single HTTP client.
    return revision
