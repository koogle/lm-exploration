"""Fetch pinned dataset artifacts into an ignored directory; verify before use."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath

INDEX = Path(__file__).with_name("pr13-artifacts.json")


def read_index(path=INDEX):
    index = json.loads(Path(path).read_text())
    if index["repo_type"] != "dataset" or not re.fullmatch(r"[0-9a-f]{40}", index["revision"]):
        raise ValueError("Artifacts require an immutable dataset revision")
    for name in index["files"]:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe artifact path")
    return index


def verify(path, entry):
    data = Path(path).read_bytes()
    if len(data) != entry["size_bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return Path(path)


def artifact_path(folder, name, index_path=INDEX):
    """Resolve an explicitly downloaded artifact; never implicitly access the network."""
    index = read_index(index_path)
    return verify(Path(folder) / name, index["files"][name])


def fetch_artifacts(folder, index_path=INDEX):
    from huggingface_hub import hf_hub_download

    index = read_index(index_path)
    for name, entry in index["files"].items():
        source = hf_hub_download(
            repo_id=index["repo_id"],
            repo_type="dataset",
            revision=index["revision"],
            filename=entry["path"],
        )
        verify(source, entry)
        target = Path(folder) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        verify(target, entry)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pr-13"))
    args = parser.parse_args()
    index = fetch_artifacts(args.output, args.index)
    print(f"Verified {len(index['files'])} artifacts at {index['revision']} in {args.output}")


if __name__ == "__main__":
    main()
