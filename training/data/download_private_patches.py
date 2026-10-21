"""Authenticated local download of a pinned, repository-stratified patch sample.

`--sources` is a JSON list of `{"repo", "revision", "default_license"?, "files"?}`
entries; revisions must be immutable commit SHAs. Experiment manifests name the datasets.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def download(sources_path, output, per_repository=24):
    # None selects every shard in each pinned source.
    if per_repository is not None and per_repository < 1:
        raise ValueError("Shard cap must be positive or None")
    import httpx
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    for source in json.loads(Path(sources_path).read_text()):
        if not re.fullmatch(r"[a-f0-9]{40}", source["revision"]):
            raise ValueError("Source revision must be an immutable commit SHA")
        if "files" not in source:
            info = HfApi().dataset_info(
                source["repo"], revision=source["revision"], files_metadata=True
            )
            source["files"] = [
                {"path": file.rfilename, "bytes": file.size}
                for file in info.siblings
                if file.rfilename.startswith("changes/") and file.rfilename.endswith(".jsonl.gz")
            ]
        for file in source["files"]:
            repo = re.sub(r"(?:-part\d+)?-\d+\.jsonl\.gz$", "", file["path"].rsplit("/", 1)[-1])
            groups[(source["repo"], repo)].append((source, file))
    selected = []
    for records in groups.values():
        selected.extend(
            sorted(records, key=lambda pair: hashlib.sha256(pair[1]["path"].encode()).hexdigest())[
                :per_repository
            ]
        )
    (output / "selection.json").write_text(
        json.dumps(
            {
                "selected_shards": len(selected),
                "available_shards": sum(map(len, groups.values())),
                "per_repository_cap": per_repository,
                "sources": [
                    {k: v for k, v in source.items() if k != "files"}
                    for source in json.loads(Path(sources_path).read_text())
                ],
            },
            indent=2,
        )
    )

    def fetch(item):
        source, file = item
        record = {"dataset": source["repo"], "revision": source["revision"], "file": file["path"]}
        if source.get("default_license"):
            record["default_license"] = source["default_license"]
        try:
            local = hf_hub_download(
                source["repo"], file["path"], repo_type="dataset", revision=source["revision"]
            )
            with open(local, "rb") as handle:
                sha = hashlib.file_digest(handle, "sha256").hexdigest()
            record.update(local_path=local, sha256=sha)
        except (httpx.HTTPError, HfHubHTTPError, LocalEntryNotFoundError, OSError) as error:
            record["error"] = type(error).__name__
        return record

    with (
        (output / "downloaded.jsonl").open("w") as handle,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        futures = [pool.submit(fetch, item) for item in selected]
        for n, future in enumerate(as_completed(futures), 1):
            record = future.result()
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if n % 10 == 0:
                print(f"Processed {n}/{len(selected)} source shards", flush=True)
    (output / "complete").touch()


if __name__ == "__main__":
    import truststore

    truststore.inject_into_ssl()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    download(args.sources, args.output)
