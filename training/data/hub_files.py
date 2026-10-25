"""Rows of dataset files pinned to a Hugging Face commit: Parquet, JSON Lines or JSON."""

import json
from pathlib import Path

from .cached_shards import parquet_rows


def download(repo, revision, name):
    """Local path of one file of a dataset repository at a full commit hash."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision))


def read_rows(path):
    """All rows of a Parquet, JSON Lines or JSON-array file, in file order."""
    path = Path(path)
    if path.suffix == ".parquet":
        return [row for _, row in parquet_rows(path)]
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{path.name} is not a JSON array of rows")
        return rows
    raise ValueError(f"Unsupported dataset file: {path.name}")


def pinned_rows(repo, revision, name):
    return read_rows(download(repo, revision, name))
