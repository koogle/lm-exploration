"""Resumable exact token counts; corpus shards are read-only and revision-pinned."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import EntryNotFoundError

from hf_store import (
    COMPONENT,
    RECORD_SCHEMA,
    HFStore,
    commit_busy,
    json_bytes,
    retry_read,
    transient_error,
)

ENCODING = "o200k_base"


class AuditCache:
    """Writes only its token path, never the collection/controller progress files."""

    def __init__(self, store, source_id):
        if not COMPONENT.fullmatch(source_id):
            raise ValueError("Unsafe source identifier")
        self.store = store
        self.path = f"tokens/{source_id}.json"
        self.previous = None
        self.revision = None

    def _head(self):
        return retry_read(
            self.store.api.repo_info, repo_id=self.store.repo_id, repo_type="dataset"
        ).sha

    def _body(self, revision):
        try:
            return self.store._download_bytes(self.path, revision)
        except EntryNotFoundError:
            return None

    def load(self):
        self.revision = self._head()
        self.previous = self._body(self.revision)
        return json.loads(self.previous) if self.previous is not None else None

    def save(self, value):
        body = json_bytes(value)
        for attempt in range(3):
            try:
                result = self.store.api.create_commit(
                    repo_id=self.store.repo_id,
                    repo_type="dataset",
                    parent_commit=self.revision,
                    commit_message=f"Checkpoint exact {ENCODING} token audit",
                    operations=[CommitOperationAdd(path_in_repo=self.path, path_or_fileobj=body)],
                )
                self.revision, self.previous = result.oid, body
                return
            except Exception as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                if not (transient_error(error) or commit_busy(error) or status in {409, 412}):
                    raise
                head = self._head()
                current = self._body(head)
                if current == body:  # An acknowledged response may have been lost.
                    self.revision, self.previous = head, body
                    return
                if current != self.previous or attempt == 2:
                    raise  # A different auditor changed this cache; never overwrite it.
                self.revision = head  # Unrelated control paths may safely advance.
                time.sleep(attempt + 1)


def measure_shard(body, shard, encoding):
    rows = pq.read_table(io.BytesIO(body)).to_pylist()
    if len(rows) != shard["rows"]:
        raise ValueError("Shard row count differs from pinned progress")
    normalized = [{field: row.get(field) for field in RECORD_SCHEMA.names} for row in rows]
    if hashlib.sha256(json_bytes(normalized)).hexdigest() != shard["sha256"]:
        raise ValueError("Shard content differs from pinned progress hash")
    if any(not isinstance(row["text"], str) for row in rows):
        raise ValueError("Cannot count a non-string document text")
    return {
        "path": shard["path"],
        "rows": len(rows),
        "characters": sum(len(row["text"]) for row in rows),
        "tokens": sum(len(encoding.encode(row["text"], disallowed_special=())) for row in rows),
        "parquet_sha256": hashlib.sha256(body).hexdigest(),
    }


def run_audit(source_id, corpus, cache_store, encoding, max_shards=100, work_seconds=600):
    if max_shards < 1 or work_seconds <= 0:
        raise ValueError("Positive shard and time budgets required")
    if encoding.name != ENCODING:
        raise ValueError("Expected o200k_base tokenizer")
    deadline = time.monotonic() + work_seconds
    state, revision = corpus.load()
    cache = AuditCache(cache_store, source_id)
    value = cache.load()
    identity = {"schema_version": 1, "encoding": ENCODING, "repo_id": corpus.repo_id}
    if value is None:
        value = {**identity, "shards": {}}
    elif any(value.get(key) != expected for key, expected in identity.items()):
        raise ValueError("Incompatible token audit cache")
    shards = state["shards"]
    if len({item["path"] for item in shards}) != len(shards):
        raise ValueError("Duplicate paths in pinned shard inventory")
    processed = 0

    def checkpoint():
        measured = []
        for shard in shards:
            entry = value["shards"].get(shard["sha256"])
            if entry is not None:
                if entry["path"] != shard["path"] or entry["rows"] != shard["rows"]:
                    raise ValueError("Cached shard identity mismatch")
                measured.append(entry)
        summary = {
            "source_id": source_id,
            "repo_id": corpus.repo_id,
            "source_revision": revision,
            "encoding": ENCODING,
            "counting_rule": "Sum of independently encoded text fields; no separators or EOS",
            "observed_at": datetime.now(UTC).isoformat(),
            "total_shards": len(shards),
            "counted_shards": len(measured),
            "total_rows": sum(shard["rows"] for shard in shards),
            "counted_rows": sum(entry["rows"] for entry in measured),
            "counted_characters": sum(entry["characters"] for entry in measured),
            "exact_tokens_counted": sum(entry["tokens"] for entry in measured),
            "complete_for_revision": len(measured) == len(shards),
            "new_shards_this_run": processed,
        }
        value["summary"] = summary
        cache.save(value)
        return summary

    for shard in shards:
        if shard["sha256"] in value["shards"]:
            continue
        if processed >= max_shards or time.monotonic() >= deadline:
            break
        body = corpus._download_bytes(shard["path"], revision)
        value["shards"][shard["sha256"]] = measure_shard(body, shard, encoding)
        processed += 1
        if processed % 10 == 0:
            checkpoint()
    return checkpoint()


def audit_source(
    source_id, max_shards=100, work_seconds=600, control_repo="jfrick/eu-corpus-control"
):
    mapping = json.loads(Path(__file__).with_name("goal_pipeline_sources.json").read_text())
    repo_id = mapping["sources"][source_id]["repo_id"]
    if control_repo == repo_id:
        raise ValueError("Token audit cache must be separate from the corpus repository")
    # A tokenizer download failure aborts; no character-based estimate is substituted.
    encoding = tiktoken.get_encoding(ENCODING)
    return run_audit(
        source_id, HFStore(repo_id), HFStore(control_repo), encoding, max_shards, work_seconds
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id")
    parser.add_argument("--max-shards", type=int, default=100)
    parser.add_argument("--work-seconds", type=float, default=600)
    parser.add_argument("--control-repo", default="jfrick/eu-corpus-control")
    args = parser.parse_args()
    print(json.dumps(audit_source(**vars(args)), indent=2))
