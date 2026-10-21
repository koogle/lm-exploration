"""Stop-era archive of the old sweep, excluding the active phased student."""

import modal

from distillation.modal_app import image
from distillation.modal_vllm import volume

app = modal.App("lm-distillation-old-sweep-archive", include_source=False)
ROOT = "python-distillation-500m-20261021-01"


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("distillation-huggingface")],
    env={"HF_HOME": "/tmp/hf-archive-sweep", "HF_XET_CACHE": "/tmp/xet-archive-sweep"},
    cpu=4,
    memory=8192,
    timeout=86400,
    retries=2,
    max_containers=1,
    volumes={"/distill": volume},
)
def archive_stopped_sweep():
    import hashlib
    import json
    import tarfile
    from datetime import UTC, datetime
    from pathlib import Path

    from huggingface_hub import HfApi, hf_hub_download

    from distillation.scale_data import atomic_json

    volume.reload()
    root = Path("/distill") / ROOT
    output = root / "stopped-archive"
    output.mkdir(exist_ok=True)
    api = HfApi()
    repo = f"{api.whoami()['name']}/python-math-500m-sweep-stopped-20261021"
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)

    def sha(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    index = {
        "format": "stopped-sweep-archive-v1",
        "run_id": ROOT,
        "observed_at": datetime.now(UTC).isoformat(),
        "repo_id": repo,
        "policy": "Old sweep stopped by user. Original volume retained. Active shared-qwen-215m-4p5b excluded.",
        "files": [],
        "producers": {},
        "students": {},
    }
    # All four producers, including uneven tails that available_chunks omits.
    for worker in range(4):
        folder = root / f"producer-{worker}"
        progress = json.loads((folder / "progress.json").read_text())
        for chunk in progress["chunks"]:
            if sha(folder / chunk["file"]) != chunk["sha256"]:
                raise ValueError(f"Corrupt source shard: {worker}/{chunk['file']}")
        index["producers"][folder.name] = {
            k: v for k, v in progress.items() if k != "chunks"
        }
        path = output / f"{folder.name}.tar"
        with tarfile.open(path, "w") as tar:
            tar.add(folder, arcname=folder.name)
        descriptor = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        }
        result = api.upload_file(
            repo_id=repo,
            repo_type="dataset",
            path_in_repo=path.name,
            path_or_fileobj=str(path),
            commit_message=f"Preserve all {folder.name} data and audits",
        )
        descriptor["revision"] = result.oid
        index["files"].append(descriptor)
        atomic_json(output / "index.json", index)
        volume.commit()
        print(json.dumps(descriptor), flush=True)

    path = output / "checkpoints-and-metadata.tar"
    with tarfile.open(path, "w") as tar:
        for item in root.iterdir():
            if item.is_file():
                tar.add(item, arcname=item.name)
        for name in ("fresh-27m", "fresh-48m", "existing-216m"):
            folder = root / name
            index["students"][name] = json.loads((folder / "progress.json").read_text())
            for item in folder.rglob("*"):
                if item.is_file() and (
                    item.suffix != ".pt" or item.name == "latest.pt"
                ):
                    tar.add(item, arcname=str(item.relative_to(root)))
    result = api.upload_file(
        repo_id=repo,
        repo_type="dataset",
        path_in_repo=path.name,
        path_or_fileobj=str(path),
        commit_message="Preserve latest resumable student checkpoints and diagnostics",
    )
    index["files"].append(
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha(path),
            "revision": result.oid,
        }
    )
    index["accepted_tokens"] = sum(p["tokens"] for p in index["producers"].values())
    index["accepted_records"] = sum(
        p["accepted_records"] for p in index["producers"].values()
    )
    atomic_json(output / "index.json", index)
    (output / "README.md").write_text(
        "# Stopped Python/math distillation sweep\n\n"
        "Private archive of Qwen3-8B generated problems, code and explanations. "
        "Producer tar files preserve all accepted chunk JSONL.gz files (text, ids, "
        "qwen_ids), raw audits, progress and generation provenance, including "
        "uneven producer tails. Syntax checked; semantic correctness unverified. "
        "Token totals use the original frozen tokenizer, not Qwen.\n\n"
        "checkpoints-and-metadata.tar preserves latest resumable checkpoints, "
        "metrics, validation and generations for the 27M, 48M and old 215.9M pilot. "
        "Other historical weight snapshots remain on the original Modal volume. "
        "The active 215.9M phased run is excluded. File SHA256 hashes and immutable "
        "upload revisions are in index.json. No source data was deleted.\n"
    )
    result = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(output),
        allow_patterns=["index.json", "README.md"],
        commit_message="Finalize stopped sweep archive index",
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            revision=result.oid,
            filename="index.json",
        )
    )
    if sha(downloaded) != sha(output / "index.json"):
        raise ValueError("Remote index verification failed")
    receipt = {
        "repo_id": repo,
        "repo_type": "dataset",
        "revision": result.oid,
        "index_path": "index.json",
        "index_sha256": sha(downloaded),
        "accepted_tokens": index["accepted_tokens"],
        "accepted_records": index["accepted_records"],
        "status": "archived",
    }
    atomic_json(root / "stopped-archive-receipt.json", receipt)
    volume.commit()
    return receipt
