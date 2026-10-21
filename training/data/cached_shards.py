"""Cache compressed JSON shards locally so reconnects do not redownload them."""

import gzip
import json
from pathlib import Path

from .simple_llm import MAX_LINE_BYTES, stream_rows


class CachedShards:
    def __init__(self, folder="/tmp/preparation-shards"):
        self.folder = Path(folder)
        self.last = {}

    def __call__(self, source, shard, start=0):
        if source["format"] != "json.gz":
            yield from stream_rows(source, shard, start)
            return
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
        with gzip.open(path, "rb") as handle:
            index = 0
            while line := handle.readline(MAX_LINE_BYTES + 1):
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError("Upstream JSON record exceeds memory bound")
                if index >= start:
                    yield index, json.loads(line)
                index += 1
