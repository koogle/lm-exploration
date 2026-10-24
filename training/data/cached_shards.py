"""Download whole upstream shards to local disk, one per active source.

A full download is much faster than streaming small pieces: a 92 MB ClimbMix
Parquet shard downloads and parses at about 36M characters/s, against about
1.6M/s for HTTP range reads. Reconnects reuse the cached file.
"""

import gzip
import json
from pathlib import Path

from .simple_llm import MAX_LINE_BYTES


class CachedShards:
    def __init__(self, folder="/tmp/preparation-shards"):
        self.folder = Path(folder)
        self.last = {}

    def __call__(self, source, shard, start=0):
        from huggingface_hub import hf_hub_download

        # Per-source directories bound cached storage to one compressed shard per
        # active source. Local_dir avoids a second retained copy in the Hub cache.
        directory = self.folder / source["name"]
        old = self.last.get(source["name"])
        if old is not None and old != shard:
            old_path = directory / old
            if old_path.exists():
                old_path.unlink()
        path = hf_hub_download(
            repo_id=source["repo"],
            repo_type="dataset",
            revision=source["revision"],
            filename=shard,
            local_dir=directory,
        )
        self.last[source["name"]] = shard
        if source["format"] == "parquet":
            yield from parquet_rows(path, start)
            return
        with gzip.open(path, "rb") as handle:
            index = 0
            while line := handle.readline(MAX_LINE_BYTES + 1):
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError("Upstream JSON record exceeds memory bound")
                if index >= start:
                    yield index, json.loads(line)
                index += 1


def parquet_rows(path, start=0):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    offset = 0
    for group in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(group).num_rows
        if offset + size <= start:
            offset += size
            continue
        for batch in parquet.iter_batches(batch_size=1024, row_groups=[group]):
            for row in batch.to_pylist():
                if offset >= start:
                    yield offset, row
                offset += 1
