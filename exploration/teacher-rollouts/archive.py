"""Archive an explicit downloaded run to an existing HF dataset; verify every byte."""

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core import atomic_json


def main():
    from huggingface_hub import HfApi, hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if not (root / "manifest.json").exists():
        raise ValueError("expected an explicit downloaded run directory")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("symlinks are not archival inputs")
    before = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    commit = HfApi().upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(root),
        path_in_repo=args.prefix,
        commit_message=f"Archive teacher rollout {root.name}",
    )
    def verify(item):
        relative, checksum = item
        remote = f"{args.prefix.rstrip('/')}/{relative}"
        downloaded = Path(
            hf_hub_download(
                args.repo_id, remote, repo_type="dataset", revision=commit.oid
            )
        )
        if hashlib.sha256(downloaded.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"archive verification failed: {remote}")
        return {"path": remote, "sha256": checksum}

    # Verify independent immutable objects concurrently; only publish a receipt
    # after every download matches. This avoids serial HEAD/GET latency for runs
    # with hundreds of individual raw/validated records.
    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(pool.map(verify, before.items()))
    atomic_json(
        args.index,
        {
            "repo_id": args.repo_id,
            "repo_type": "dataset",
            "revision": commit.oid,
            "files": entries,
        },
    )


if __name__ == "__main__":
    main()
