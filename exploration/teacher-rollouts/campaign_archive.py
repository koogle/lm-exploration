"""Incremental durable archive of native Qwen3.5 trajectories and sparse targets."""

import hashlib
import json
import time
from pathlib import Path

from core import atomic_json


def archive(root, workers, deadline):
    import modal
    from huggingface_hub import CommitOperationAdd, HfApi

    volume = modal.Volume.from_name("lm-exploration-teacher-rollouts")
    api = HfApi()
    repo = f"{api.whoami()['name']}/qwen35-python-math-trajectories-20261022"
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    root = Path(root)
    index_path = root / "archive-index.json"
    index = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {"files": [], "generated_tokens": 0, "accepted_tokens": 0, "records": 0}
    )
    seen = {x["path"] for x in index["files"]}
    while time.time() < deadline + 3600:
        volume.reload()
        pending = []
        for worker in range(workers):
            path = root / f"producer-{worker}" / "progress.json"
            if not path.exists():
                continue
            state = json.loads(path.read_text())
            for chunk in state["chunks"]:
                remote = f"data/{root.name}/producer-{worker}/{chunk['file']}"
                if remote not in seen:
                    pending.append(
                        (root / f"producer-{worker}" / chunk["file"], remote, chunk)
                    )
        if not pending:
            if time.time() > deadline + 60:
                return {"status": "archived", "repo_id": repo, **index}
            time.sleep(30)
            continue
        operations = []
        for local, remote, chunk in pending[:64]:
            if hashlib.sha256(local.read_bytes()).hexdigest() != chunk["sha256"]:
                raise ValueError("Shard corruption")
            operations.append(
                CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
            )
            index["files"].append({"path": remote, **chunk})
            seen.add(remote)
            for key in ["generated_tokens", "accepted_tokens", "records"]:
                index[key] += chunk[key]
        data = (json.dumps(index, indent=2) + "\n").encode()
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"runs/{root.name}/index.json", path_or_fileobj=data
            )
        )
        card = b"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/**/*.jsonl.gz
---
# Qwen3.5 Python/math trajectories

Native pinned Qwen3.5-9B prompt/output token IDs and raw top-64 log probabilities,
selected-token log probability and residual probability mass. Top-k is not
renormalized. Filter accepted=true for JSON/syntax-checked examples; semantic
correctness is not established. Benchmark fixtures are excluded. These token IDs
are incompatible with the old shared-Qwen student vocabulary. Use runs/<campaign>/index.json at
an immutable commit for SHA256 hashes. Generated data includes rejected outputs
for audit. The old Qwen3-8B corpus remains separate.
"""
        operations.append(
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card)
        )
        manifest = root / "campaign.json"
        if manifest.exists():
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"runs/{root.name}/campaign.json",
                    path_or_fileobj=str(manifest),
                )
            )
        tokenizers = list(root.glob("producer-*/tokenizer/tokenizer.json"))
        if tokenizers and not index.get("tokenizer_archived"):
            for path in tokenizers[0].parent.iterdir():
                if path.is_file():
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo=f"runs/{root.name}/tokenizer/" + path.name,
                            path_or_fileobj=str(path),
                        )
                    )
            index["tokenizer_archived"] = True
        commit = api.create_commit(
            repo_id=repo,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Archive {index['generated_tokens']} teacher targets",
        )
        atomic_json(index_path, index)
        atomic_json(
            root / "archive-progress.json",
            {
                "repo_id": repo,
                "revision": commit.oid,
                "index_sha256": hashlib.sha256(data).hexdigest(),
                **{
                    k: index[k]
                    for k in ["generated_tokens", "accepted_tokens", "records"]
                },
            },
        )
        volume.commit()
    return {"status": "archive_deadline", "repo_id": repo}
