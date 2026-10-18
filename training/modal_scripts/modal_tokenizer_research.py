"""CPU-only, bounded Modal tokenizer learning curve; no ledger prerequisite."""

import json
import re
from pathlib import Path

import modal

app = modal.App("lm-exploration-tokenizer-research")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data", create_if_missing=True)
results_volume = modal.Volume.from_name("lm-exploration-training", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "tokenizers==0.22.2", "requests==2.34.2", "pyarrow==26.0.0", "huggingface-hub==1.33.0"
    )
    .env({"PYTHONPATH": "/project", "RAYON_NUM_THREADS": "8"})
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer", "/project/tokenizer")
    .add_local_dir("tests", "/project/tests", ignore=["__pycache__"])
)


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=7200,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume},
)
def collect(run_id: str):
    import time

    from training_data.scholarly import load_recipe

    from tokenizer.tokenizer_research import CorpusBuilder, load_spec

    spec = load_spec("/project/tokenizer/research/learning-curve-v1.json")
    recipe = load_recipe("/project/" + spec["source_recipe"])
    root = Path("/data/tokenizer-research") / run_id
    builder = CorpusBuilder(root, spec, recipe)
    last_commit = time.monotonic()

    def checkpoint():
        nonlocal last_commit
        # SQLite commits batches locally. Uploading the growing database on every
        # batch makes network checkpoint latency dominate corpus collection.
        if time.monotonic() - last_commit >= 60:
            data_volume.commit()
            last_commit = time.monotonic()

    try:
        if (root / "frozen/manifest.json").exists():
            from tokenizer.tokenizer_research import verify_corpus

            return verify_corpus(root / "frozen")
        # A later cross-source duplicate can invalidate an earlier source's quota.
        for _ in range(20):
            for source in recipe["sources"]:
                name = source["name"]
                if not builder.complete(name):
                    print(f"Collecting {name}: {builder.counts(name)}", flush=True)
                    builder.collect(name, checkpoint=checkpoint)
                    data_volume.commit()
                    print(f"Collected {name}: {builder.counts(name)}", flush=True)
            if all(builder.complete(s["name"]) for s in recipe["sources"]):
                result = builder.freeze()
                data_volume.commit()
                return result
        raise ValueError("Collection bound reached without full quotas")
    finally:
        builder.close()
        data_volume.commit()


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=7200,
    retries=0,
    max_containers=3,
    volumes={"/data": data_volume, "/results": results_volume},
)
def candidate(run_id: str, size_mib: int, vocab_size: int):
    import subprocess
    import sys

    data_volume.reload()
    root = Path("/results/tokenizer-research") / run_id / f"{size_mib}mib-{vocab_size}"
    if root.exists():
        raise ValueError("Candidate path exists; inspect results instead of overwriting")
    root.parent.mkdir(parents=True, exist_ok=True)
    # Fresh process isolates peak RSS and native tokenizer allocations per candidate.
    script = """
import sys
from tokenizer.tokenizer_research import train_candidate
train_candidate(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),
                '/project/tokenizer/scholarly-code-v1/config.json',execution='modal-cpu')
"""
    print(f"Training {size_mib} MiB / {vocab_size} vocabulary", flush=True)
    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                f"/data/tokenizer-research/{run_id}/frozen",
                str(root),
                str(size_mib),
                str(vocab_size),
            ],
            check=True,
            timeout=7000,
        )
        result = json.loads((root / "result.json").read_text())
        print(
            f"Finished {size_mib} MiB / {vocab_size}: "
            f"{result['metrics']['bytes_per_token']:.4f} bytes/token",
            flush=True,
        )
        return result
    finally:
        results_volume.commit()


@app.function(image=image, cpu=2, memory=4096, timeout=600, retries=0, max_containers=1)
def smoke():
    import subprocess
    import sys

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "/project/tests",
            "-p",
            "test_tokenizer_research.py",
            "-v",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    report = {
        "returncode": run.returncode,
        "stdout": run.stdout,
        "stderr": run.stderr,
        "execution": "modal-cpu",
        "cpu": 2,
        "memory_mib": 4096,
    }
    print(run.stderr, flush=True)
    if run.returncode:
        raise RuntimeError(run.stderr)
    return report


@app.local_entrypoint()
def main(run_id: str, phase: str = "all", size_mib: int = 0, vocab_size: int = 0):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Use a simple versioned run ID")
    if phase not in ("smoke", "collect", "train", "all", "candidate"):
        raise ValueError("Unknown phase")
    spec = json.loads(Path("tokenizer/research/learning-curve-v1.json").read_text())
    out = Path("results/tokenizer-learning-curve") / run_id
    out.mkdir(parents=True, exist_ok=True)
    if phase in ("smoke", "all"):
        (out / "modal-smoke.json").write_text(json.dumps(smoke.remote(), indent=2) + "\n")
    if phase in ("collect", "all"):
        manifest = collect.remote(run_id)
        (out / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if phase in ("train", "all", "candidate"):
        matrix = [(run_id, s, v) for s in spec["sizes_mib"] for v in spec["vocab_sizes"]]
        if phase == "candidate":
            if size_mib not in spec["sizes_mib"] or vocab_size not in spec["vocab_sizes"]:
                raise ValueError("Candidate not in matrix")
            matrix = [(run_id, size_mib, vocab_size)]
        for result in candidate.starmap(matrix, order_outputs=False):
            name = f"{result['size_mib']}mib-{result['requested_vocab_size']}.json"
            (out / name).write_text(json.dumps(result, indent=2) + "\n")
