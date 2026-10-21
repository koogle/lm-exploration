"""Publish an immutable committed-code snapshot alongside durable HF control state."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[3]
SCOPE = "training/data/eu"


def snapshot():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar.gz", revision, SCOPE], cwd=ROOT
    )
    if len(archive) > 32 * 1024 * 1024:
        raise ValueError("Code backup exceeds 32 MiB; review tracked contents")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        files = [entry.name for entry in source if entry.isfile()]
    for name in files:
        path = Path(name)
        if (
            "artifacts" in path.parts
            or any(part in {".env", ".venv", ".git"} for part in path.parts)
            or path.suffix.lower() in {".pem", ".key", ".parquet"}
        ):
            raise ValueError(f"Unexpected local/bulk/credential file in snapshot: {name}")
    manifest = {
        "git_commit": revision,
        "scope": SCOPE,
        "git_committed_at": subprocess.check_output(
            ["git", "show", "-s", "--format=%cI", revision], cwd=ROOT, text=True
        ).strip(),
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "archive_bytes": len(archive),
        "files": files,
        "restore": "Verify archive_sha256, then extract tar.gz; paths start at training/data/eu/.",
        "exclusions": "Only committed Git files; no ignored artifacts, credentials or uncommitted edits.",
    }
    return revision, archive, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="jfrick/eu-corpus-control")
    args = parser.parse_args()
    revision, archive, manifest = snapshot()
    api = HfApi()
    info = api.repo_info(args.repo, repo_type="dataset")
    if not info.private:
        raise ValueError("Code backup destination must remain private")
    prefix = f"code/{revision}"
    # Touch only code paths. Never upload/replace progress.json or the dataset card.
    result = api.create_commit(
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=f"Back up collector code at {revision[:12]}",
        operations=[
            CommitOperationAdd(path_in_repo=f"{prefix}.tar.gz", path_or_fileobj=archive),
            CommitOperationAdd(
                path_in_repo=f"{prefix}.json",
                path_or_fileobj=(json.dumps(manifest, indent=2) + "\n").encode(),
            ),
        ],
    )
    downloaded = api.hf_hub_download(
        repo_id=args.repo,
        repo_type="dataset",
        filename=f"{prefix}.tar.gz",
        revision=result.oid,
    )
    if hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Uploaded code snapshot failed checksum verification")
    print(
        json.dumps(
            {
                "repo": args.repo,
                "git_commit": revision,
                "hf_revision": result.oid,
                "files": len(manifest["files"]),
                "bytes": len(archive),
                "verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
