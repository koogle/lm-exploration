"""Commit reusable accepted examples and a pinned index to a private HF dataset."""

import gzip
import hashlib
import json
import time

from distillation.scale_data import atomic_json, available_chunks, read_chunk


def archive(root, name, fresh_root, commit, reload_volume, seconds=82000):
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    owner = api.whoami()["name"]
    repo = f"{owner}/python-math-qwen3-distillation-20261021"
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    output = root / name
    local = output / "dataset-archive"
    local.mkdir(exist_ok=True)
    path = local / "index.json"
    index = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "format": "qwen-python-math-jsonl-v1",
            "repo_id": repo,
            "repo_type": "dataset",
            "chunks_archived": 0,
            "qwen_tokens": 0,
            "records": 0,
            "parts": [],
            "teacher": "Qwen/Qwen3-8B",
            "teacher_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "tokenizer_sha256": "a63080b4b30dc861fe1b0e420c53a241304dc81aff3d9bb315e6e33e2a1ab794",
            "verification": "Syntax checked; semantics unverified. No corpus replay; semantic and cross-worker duplicates possible.",
        }
    )
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        reload_volume()
        plan_path = output / "dataset-plan.json"
        if not plan_path.exists():
            time.sleep(30)
            continue
        plan = json.loads(plan_path.read_text())
        ordered = [(root, item) for item in plan["initial_chunks"]]
        ordered.extend((fresh_root, item) for item in available_chunks(fresh_root))
        start = index["chunks_archived"]
        progress = json.loads((output / "progress.json").read_text())
        tail = ordered[start : start + 128]
        if not tail or (
            len(tail) < 64
            and index["parts"]
            and progress["tokens"] < progress["target"]
        ):
            time.sleep(60)
            continue
        part = local / f"part-{len(index['parts']):06d}.jsonl.gz"
        records = tokens = 0
        sources = []
        with gzip.open(part, "wt") as handle:
            for corpus, item in tail:
                sources.append({"corpus": corpus.name, "chunk": item})
                for row in read_chunk(corpus, item):
                    row["source_corpus"] = corpus.name
                    handle.write(json.dumps(row) + "\n")
                    records += 1
                    tokens += len(row["qwen_ids"])
        descriptor = {
            "path": f"data/{part.name}",
            "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
            "records": records,
            "qwen_tokens": tokens,
            "first_chunk": start,
            "end_chunk": start + len(tail),
            "sources": sources,
        }
        new_index = {
            **index,
            "parts": index["parts"] + [descriptor],
            "chunks_archived": start + len(tail),
            "qwen_tokens": index["qwen_tokens"] + tokens,
            "records": index["records"] + records,
            "student_progress": progress,
            "dataset_plan": plan,
        }
        payload = (json.dumps(new_index, indent=2) + "\n").encode()
        card = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.jsonl.gz
---
# Qwen3 Python/math distillation dataset

Fresh Qwen3-8B problems, Python code and explanations, with archived shared-Qwen
`qwen_ids` (including EOS), original-tokenizer `ids`, text, content hash and prompt
seed provenance. Syntax checked, not execution-verified. No intentional corpus
replay; exact deduplication is per producer, not global or semantic.

Use `index.json` at a pinned commit for shard SHA256 checksums, ordered source
chunks, tokenizer/teacher identity and the student checkpoint's consumed-token
prefix. The dataset may contain a small unused tail beyond the 4.5B training
budget. Save the commit revision returned by the archive; do not assume `main`
is immutable. Private research dataset. Training uses all-token CE, context 512.
"""
        result = api.create_commit(
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"Archive {new_index['qwen_tokens']} accepted Qwen targets",
            operations=[
                CommitOperationAdd(
                    path_in_repo=descriptor["path"], path_or_fileobj=str(part)
                ),
                CommitOperationAdd(path_in_repo="index.json", path_or_fileobj=payload),
                CommitOperationAdd(
                    path_in_repo="README.md", path_or_fileobj=card.encode()
                ),
            ],
        )
        index = new_index
        atomic_json(path, index)
        atomic_json(
            output / "dataset-archive.json",
            {
                "repo_id": repo,
                "repo_type": "dataset",
                "revision": result.oid,
                "index_path": "index.json",
                "index_sha256": hashlib.sha256(payload).hexdigest(),
                "qwen_tokens": index["qwen_tokens"],
                "records": index["records"],
                "chunks_archived": index["chunks_archived"],
            },
        )
        commit()
        print(
            json.dumps(
                {
                    "dataset_repo": repo,
                    "revision": result.oid,
                    "qwen_tokens": index["qwen_tokens"],
                }
            ),
            flush=True,
        )
    return {"status": "continue", "repo_id": repo}
