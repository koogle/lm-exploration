"""Deployed, parallel, bounded full-file Python code preparation; never allocates GPUs."""

import json
from pathlib import Path

import modal

app = modal.App("simple-llm-python-code-supplement")
volume = modal.Volume.from_name("lm-exploration-mixture-data")
ROOT = "/data/python-code-supplement-20261021-v1"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tokenizers==0.22.2", "huggingface_hub==0.36.0", "requests")
    .env({"PYTHONPATH": "/project", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_XET": "1"})
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer/structured-code-v1", "/project/tokenizer/structured-code-v1")
)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    retries=1,
    max_containers=32,
    volumes={"/data": volume},
)
def shard(filename: str):
    print(json.dumps({"shard_started": filename}), flush=True)
    import gzip
    import hashlib
    import shutil
    import tempfile

    from huggingface_hub import hf_hub_download
    from training_data.python_code_supplement import REPOSITORY, REVISION, select_document
    from training_data.python_experiment import Corpus

    if not filename.startswith("file-") or not filename.endswith(".json.gz"):
        raise ValueError("Unexpected source shard")
    folder = Path(ROOT) / Path(filename).name.removesuffix(".json.gz")
    done = folder / "complete.json"
    if done.exists():
        return json.loads(done.read_text())
    # SQLite transactions stay on local disk. A failed parser/download must never
    # leave a connection open against the shared volume for a subsequent retry.
    with tempfile.TemporaryDirectory(prefix="python-code-") as scratch:
        local = Path(scratch)
        corpus = Corpus(local)
        try:
            path = hf_hub_download(REPOSITORY, filename, repo_type="dataset", revision=REVISION)
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                while block := handle.read(8 * 1024 * 1024):
                    digest.update(block)
            scanned = 0
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    scanned += 1
                    try:
                        record = json.loads(line)
                    except (ValueError, TypeError):
                        corpus.reject("malformed_json")
                        continue
                    selected, reason = select_document(record)
                    if selected is None:
                        corpus.reject(reason)
                        continue
                    text, category, parent, provenance = selected
                    provenance.update(shard=filename, shard_sha256=digest.hexdigest())
                    corpus.add(text, category, parent, provenance)
                    if scanned % 5000 == 0:
                        corpus.report("code_shard_preparing")
            result = corpus.report("code_shard_complete")
            result.update(shard=filename, rows_scanned=scanned, source_sha256=digest.hexdigest())
        finally:
            corpus.db.close()
        # Publish the completion marker last; interrupted copies are never merged.
        folder.mkdir(parents=True, exist_ok=True)
        for suffix in ("-journal", "-wal", "-shm"):
            (folder / ("raw.sqlite" + suffix)).unlink(missing_ok=True)
        for source in local.iterdir():
            if source.is_file():
                pending = folder / (source.name + ".pending")
                shutil.copyfile(source, pending)
                pending.replace(folder / source.name)
        done.write_text(json.dumps(result, indent=2))
        volume.commit()
    print(
        json.dumps(
            {
                "shard": filename,
                "rows": scanned,
                "capacity": result["categories"],
            }
        ),
        flush=True,
    )
    return result


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=43200,
    retries=0,
    max_containers=1,
    volumes={"/data": volume},
)
def prepare(max_shards: int = 53):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import HfApi
    from training_data.python_code_supplement import REPOSITORY, REVISION

    if not 1 <= max_shards <= 53:
        raise ValueError("Preparation bound must be between 1 and 53 shards")
    files = sorted(
        (
            f
            for f in HfApi().list_repo_files(REPOSITORY, repo_type="dataset", revision=REVISION)
            if f.startswith("file-") and f.endswith(".json.gz")
        ),
        reverse=True,
    )
    volume.reload()
    root = Path(ROOT)
    root.mkdir(parents=True, exist_ok=True)
    calls_path = root / "spawned-calls.json"
    calls = json.loads(calls_path.read_text()) if calls_path.exists() else {}
    results, pending = [], []
    for filename in files[:max_shards]:
        done = root / filename.removesuffix(".json.gz") / "complete.json"
        if done.exists():
            results.append(json.loads(done.read_text()))
            continue
        if filename not in calls:
            calls[filename] = shard.spawn(filename).object_id
        pending.append(modal.FunctionCall.from_id(calls[filename]))
    calls_path.write_text(json.dumps(calls, indent=2))
    volume.commit()

    def report():
        total = sum(
            sum(v["unique_train_targets"] for v in r["categories"].values()) for r in results
        )
        volume.reload()
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "stage": "prepared_separate_shards",
                    "requires_global_merge_and_exclusions": True,
                    "train_targets_before_global_dedup": total,
                    "shards": results,
                },
                indent=2,
            )
        )
        volume.commit()
        print(
            json.dumps({"completed_shards": len(results), "candidate_targets": total}), flush=True
        )
        return total

    total = report()
    with ThreadPoolExecutor(max_workers=32) as pool:
        for future in as_completed([pool.submit(call.get) for call in pending]):
            results.append(future.result())
            total = report()
    return {"candidate_targets": total, "completed_shards": len(results)}
