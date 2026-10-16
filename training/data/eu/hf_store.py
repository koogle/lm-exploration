"""Atomic, resumable Hugging Face dataset storage shared by collection workers."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import EntryNotFoundError

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


def dataset_card(state):
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
        local = self.api.hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=path,
            revision=revision,
        )
        return json.loads(Path(local).read_text())

    def load(self):
        revision = self.api.repo_info(repo_id=self.repo_id, repo_type="dataset").sha
        try:
            path = self.api.hf_hub_download(
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
                }
            )
            known_paths.add(path)
        for path, body in (raw_files or {}).items():
            if not RAW_PATH.fullmatch(path) or not isinstance(body, bytes):
                raise ValueError(f"Invalid raw file: {path!r}")
            additions[path] = body
        for path, body in (extra_files or {}).items():
            if not INVENTORY_PATH.fullmatch(path) or not isinstance(body, bytes):
                raise ValueError(f"Invalid inventory file: {path!r}")
            json.loads(body)  # Inventories must actually be JSON.
            additions[path] = body
        additions["state/progress.json"] = json_bytes(next_state)
        additions["README.md"] = dataset_card(next_state)
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
        # Callers can keep their state in sync without an extra download. Mutation
        # occurs only after successful commit; conflicts and network failures propagate.
        state.clear()
        state.update(next_state)
        return result.oid
