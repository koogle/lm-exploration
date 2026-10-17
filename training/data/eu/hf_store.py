"""Atomic, resumable Hugging Face dataset storage shared by collection workers."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import EntryNotFoundError


def transient_error(error):
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status is not None:
        return status in {502, 503, 504}
    if isinstance(error, (httpx.TransportError, ConnectionError, TimeoutError)):
        return True
    # huggingface_hub <1 uses requests; avoid requiring it with the newer httpx SDK.
    return any(
        cls.__module__ == "requests.exceptions" and cls.__name__ in {"ConnectionError", "Timeout"}
        for cls in type(error).__mro__
    )


def commit_busy(error):
    """HF's temporary repository lock, distinct from a stale-parent conflict."""
    return (
        getattr(getattr(error, "response", None), "status_code", None) == 409
        and "another commit operation is in progress" in str(error).lower()
    )


def retry_read(operation, **kwargs):
    for attempt in range(3):
        try:
            return operation(**kwargs)
        except Exception as error:
            if not transient_error(error) or attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


STRING_FIELDS = [
    "id",
    "source_id",
    "collection_id",
    "document_id",
    "url",
    "title",
    "language",
    "document_type",
    "format",
    "published_at",
    "modified_at",
    "retrieved_at",
    "text",
    "text_sha256",
    "raw_sha256",
    "extraction_version",
    "provenance_json",
]
RECORD_SCHEMA = pa.schema(
    [(name, pa.string()) for name in STRING_FIELDS]
    + [("characters", pa.int64()), ("words", pa.int64())]
)
COMPONENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}$")
RAW_PATH = re.compile(r"^raw/[a-f0-9]{64}\.gz$")
INVENTORY_PATH = re.compile(
    r"^inventories/(?:[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}/)?[a-zA-Z0-9][a-zA-Z0-9_.-]{0,180}\.json$"
)


def empty_state():
    return {
        "schema_version": 1,
        "completed": {},
        "failures": {},
        "collections": {},
        "runs": [],
        "shards": [],
        "text_index": {},
    }


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


METRIC_FIELDS = (
    "rows",
    "characters",
    "words",
    "text_utf8_bytes",
    "parquet_bytes",
    "stored_raw_gzip_bytes",
)


def zero_metrics():
    return dict.fromkeys(METRIC_FIELDS, 0)


def text_metrics(rows, parquet_bytes):
    result = zero_metrics()
    result.update(
        rows=len(rows),
        characters=sum(len(row["text"] or "") for row in rows),
        words=sum(len((row["text"] or "").split()) for row in rows),
        text_utf8_bytes=sum(len((row["text"] or "").encode("utf-8")) for row in rows),
        parquet_bytes=parquet_bytes,
    )
    return result


def aggregate_metrics(state):
    totals, collections = zero_metrics(), {}
    for shard in state["shards"]:
        metrics = shard["metrics"]
        group = collections.setdefault(shard["collection_id"], zero_metrics())
        for field in METRIC_FIELDS:
            totals[field] += metrics[field]
            group[field] += metrics[field]
    # A raw blob can be shared by multiple shards. Attribute it to its first
    # referencing shard, so collection/shard sums equal unique stored byte totals.
    for blob in state["raw_blob_metrics"].values():
        if blob["shard_path"] is None:
            totals["stored_raw_gzip_bytes"] += blob["bytes"]
            collections.setdefault("unassigned_raw", zero_metrics())["stored_raw_gzip_bytes"] += (
                blob["bytes"]
            )
    state["metrics"] = totals
    state["collection_metrics"] = collections
    state["metrics_version"] = 1


def dataset_card(state):
    if "collection_goal" in state:
        goal = state["collection_goal"]
        config = goal.get("config", {})
        lines = [
            "---",
            "pretty_name: EU collection control",
            "---",
            "",
            "# EU collection control",
            "",
            "Private orchestration state; corpus documents live in organization repositories.",
            "Durable jobs, reservations, source metrics and history are in state/progress.json.",
            "",
            f"Target documents: {config.get('target_documents', 'not set')}",
            f"Status: {goal.get('status', 'initializing')}",
            f"Observed stored documents: {goal.get('stored_rows', 'not yet measured')}",
            f"Reserved document capacity: {goal.get('reserved_documents', 'not yet measured')}",
            "",
            "Sources: " + ", ".join(config.get("sources", [])),
        ]
        return "\n".join(lines).encode()
    collections = sorted({shard["collection_id"] for shard in state["shards"]})
    lines = ["---", "pretty_name: EU institutional corpus pilot"]
    if collections:
        lines.append("configs:")
        for collection in collections:
            lines.extend(
                [
                    f"- config_name: {json.dumps(collection)}",
                    "  data_files:",
                    "  - split: train",
                    f'    path: "data/{collection}/**/*.parquet"',
                ]
            )
    lines.extend(
        [
            "---",
            "",
            "# EU institutional corpus pilot",
            "",
            "Continuously collected institutional documents, partitioned by collection and format.",
            "Original source URLs and extraction provenance accompany each document.",
            "",
            "## License and status",
            "",
            "Source terms are mixed and have not been reviewed. No uniform license is asserted.",
            "This pilot is incomplete and has not established novelty against existing corpora.",
            "",
            "## Durable progress",
            "",
            "`state/progress.json` records completed work, failures, collection mappings and runs.",
            "Each progress update is committed atomically with its corresponding Parquet shards.",
            "Raw source snapshots live under `raw/`; discovery inventories under `inventories/`.",
            "",
            (
                f"Stored shards: {len(state['shards'])}. "
                f"Rows: {sum(shard['rows'] for shard in state['shards'])}."
            ),
            "Counts describe stored rows; later versions can share a document identifier.",
            "",
        ]
    )
    if "metrics" in state:
        lines.extend(
            [
                "## Stored corpus metrics",
                "",
                "Counts cover committed data files at this revision, not model tokens.",
                "Byte counts exclude inventories, progress, dataset cards and Git history.",
                "Shared raw blobs are counted once, attributed to their first referencing collection.",
                "",
                "| Scope | Rows | Characters | Words | Text UTF-8 bytes | Parquet bytes | Raw gzip bytes |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, metrics in [
            ("Total", state["metrics"]),
            *sorted(state["collection_metrics"].items()),
        ]:
            lines.append(
                "| "
                + label
                + " | "
                + " | ".join(f"{metrics[field]:,}" for field in METRIC_FIELDS)
                + " |"
            )
        lines.append("")
    return "\n".join(lines).encode()


class HFStore:
    def __init__(self, repo_id, api=None):
        self.repo_id = repo_id
        self.api = api if api is not None else HfApi()

    def ensure(self, private=True):
        # exist_ok preserves the visibility of existing repositories.
        self.api.create_repo(
            repo_id=self.repo_id, repo_type="dataset", private=private, exist_ok=True
        )

    def read_json(self, path, revision=None):
        if path != "state/progress.json" and not INVENTORY_PATH.fullmatch(path):
            raise ValueError(f"Invalid JSON path: {path!r}")
        local = retry_read(
            self.api.hf_hub_download,
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=path,
            revision=revision,
        )
        return json.loads(Path(local).read_text())

    def load(self):
        revision = retry_read(self.api.repo_info, repo_id=self.repo_id, repo_type="dataset").sha
        try:
            path = retry_read(
                self.api.hf_hub_download,
                repo_id=self.repo_id,
                repo_type="dataset",
                filename="state/progress.json",
                revision=revision,
            )
        except EntryNotFoundError:
            return empty_state(), revision
        state = json.loads(Path(path).read_text())
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported progress schema version")
        for key in empty_state():
            if key not in state:
                raise ValueError(f"Incomplete progress state: missing {key}")
        return state, revision

    def _download_bytes(self, path, revision):
        local = retry_read(
            self.api.hf_hub_download,
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=path,
            revision=revision,
        )
        return Path(local).read_bytes()

    def migrate_metrics(self, state, revision):
        """Backfill historical metrics locally from pinned files; caller persists them.

        Missing/corrupt source files fail loudly, never report fabricated zero totals.
        No caller state changes until every historical shard has been measured.
        """
        if state.get("metrics_version") == 1:
            return state
        migrated = copy.deepcopy(state)
        migrated["raw_blob_metrics"] = {}
        # HF metadata reports stored LFS/object lengths without fetching each blob.
        # Pin this lookup to the same revision as the measured Parquet files.
        sizes = {}
        if migrated["shards"]:
            info = retry_read(
                self.api.repo_info,
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=revision,
                files_metadata=True,
            )
            sizes = {
                sibling.rfilename: sibling.size
                for sibling in (getattr(info, "siblings", None) or [])
                if getattr(sibling, "size", None) is not None
            }
        for shard in migrated["shards"]:
            body = self._download_bytes(shard["path"], revision)
            rows = pq.read_table(io.BytesIO(body)).to_pylist()
            metrics = text_metrics(rows, len(body))
            if metrics["rows"] != shard["rows"]:
                raise ValueError(f"Historical shard row count mismatch: {shard['path']}")
            for row in rows:
                raw_hash = row.get("raw_sha256")
                if not raw_hash:
                    continue
                path = f"raw/{raw_hash}.gz"
                if not RAW_PATH.fullmatch(path):
                    raise ValueError(f"Invalid historical raw hash: {raw_hash!r}")
                if path not in migrated["raw_blob_metrics"]:
                    size = sizes.get(path)
                    if size is None:
                        size = len(self._download_bytes(path, revision))
                    migrated["raw_blob_metrics"][path] = {
                        "bytes": size,
                        "shard_path": shard["path"],
                    }
                    metrics["stored_raw_gzip_bytes"] += size
            shard["metrics"] = metrics
        aggregate_metrics(migrated)
        state.clear()
        state.update(migrated)
        return state

    def commit(
        self,
        state,
        revision,
        records,
        raw_files=None,
        extra_files=None,
        message="Update corpus and durable progress",
    ):
        if not revision:
            raise ValueError("A loaded HEAD revision is required for compare-and-swap")
        next_state = copy.deepcopy(state)
        self.migrate_metrics(next_state, revision)
        groups = defaultdict(list)
        seen = set()
        for record in records:
            for field in ("collection_id", "format"):
                if not COMPONENT.fullmatch(record[field]):
                    raise ValueError(f"Unsafe {field}: {record[field]!r}")
            identity = record["id"]
            if identity in seen:
                raise ValueError(f"Duplicate document ID in batch: {identity}")
            seen.add(identity)
            groups[(record["collection_id"], record["format"])].append(record)
        additions = {}
        known_paths = {shard["path"] for shard in next_state["shards"]}
        for (collection, fmt), rows in sorted(groups.items()):
            rows = sorted(rows, key=lambda row: row["id"])
            # Include all persisted fields so updated extraction/provenance never overwrites
            # a prior shard at the same path, even when the text remains unchanged.
            normalized = [{name: row.get(name) for name in RECORD_SCHEMA.names} for row in rows]
            digest = hashlib.sha256(json_bytes(normalized)).hexdigest()
            path = f"data/{collection}/{fmt}/part-{digest}.parquet"
            if path in known_paths:
                continue
            buffer = io.BytesIO()
            pq.write_table(
                pa.Table.from_pylist(normalized, schema=RECORD_SCHEMA), buffer, compression="zstd"
            )
            additions[path] = buffer.getvalue()
            next_state["shards"].append(
                {
                    "path": path,
                    "collection_id": collection,
                    "format": fmt,
                    "rows": len(rows),
                    "sha256": digest,
                    "metrics": text_metrics(normalized, len(additions[path])),
                }
            )
            known_paths.add(path)
        for path, body in (raw_files or {}).items():
            if not RAW_PATH.fullmatch(path) or not isinstance(body, bytes):
                raise ValueError(f"Invalid raw file: {path!r}")
            previous = next_state["raw_blob_metrics"].get(path)
            if previous and previous["bytes"] != len(body):
                raise ValueError(f"Raw blob size changed for immutable path: {path}")
            if not previous:
                next_state["raw_blob_metrics"][path] = {"bytes": len(body), "shard_path": None}
            additions[path] = body
        # Link raw bytes to the first newly created shard referencing each blob.
        for shard in next_state["shards"]:
            if shard["path"] not in additions:
                continue
            for row in groups[(shard["collection_id"], shard["format"])]:
                if not row.get("raw_sha256"):
                    continue
                path = f"raw/{row['raw_sha256']}.gz"
                blob = next_state["raw_blob_metrics"].get(path)
                if blob is None:
                    # Caller may reference a previously uploaded source blob.
                    blob = {"bytes": len(self._download_bytes(path, revision)), "shard_path": None}
                    next_state["raw_blob_metrics"][path] = blob
                if blob["shard_path"] is None:
                    blob["shard_path"] = shard["path"]
                    shard["metrics"]["stored_raw_gzip_bytes"] += blob["bytes"]
        aggregate_metrics(next_state)
        for path, body in (extra_files or {}).items():
            if not INVENTORY_PATH.fullmatch(path) or not isinstance(body, bytes):
                raise ValueError(f"Invalid inventory file: {path!r}")
            json.loads(body)  # Inventories must actually be JSON.
            additions[path] = body
        history = next_state.setdefault("metrics_history", [])
        if not history or history[-1]["metrics"] != next_state["metrics"]:
            history.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "revision_parent": revision,
                    "metrics": copy.deepcopy(next_state["metrics"]),
                }
            )
        additions["state/progress.json"] = json_bytes(next_state)
        additions["README.md"] = dataset_card(next_state)
        committed_revision = self._commit_with_recovery(additions, revision, message)
        # Callers can keep their state in sync without an extra download. Mutation
        # occurs only after successful commit; conflicts and network failures propagate.
        state.clear()
        state.update(next_state)
        return committed_revision

    def _commit_with_recovery(self, additions, revision, message):
        """Retry only unchanged-head CAS writes; resolve lost responses by exact state."""
        for attempt in range(3):
            try:
                result = self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    parent_commit=revision,
                    commit_message=message,
                    operations=[
                        CommitOperationAdd(path_in_repo=path, path_or_fileobj=body)
                        for path, body in sorted(additions.items())
                    ],
                )
                return result.oid
            except Exception as error:
                if not (transient_error(error) or commit_busy(error)):
                    raise
                try:
                    head = retry_read(
                        self.api.repo_info, repo_id=self.repo_id, repo_type="dataset"
                    ).sha
                    if head != revision:
                        persisted = self._download_bytes("state/progress.json", head)
                        if persisted == additions["state/progress.json"]:
                            return head
                except Exception as verification_error:
                    raise error from verification_error
                # Another writer advanced HEAD. Never rebase this stale state.
                if head != revision or attempt == 2:
                    raise
                time.sleep((attempt + 1) * (5 if commit_busy(error) else 1))
        raise AssertionError("unreachable")
