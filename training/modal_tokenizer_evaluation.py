"""Balanced-parent evaluation of the already fitted tokenizer grid."""

import json
import re
import time
from pathlib import Path

import modal

app = modal.App("lm-exploration-tokenizer-evaluation")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
results_volume = modal.Volume.from_name("lm-exploration-training")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "tokenizers==0.22.2", "requests==2.34.2", "pyarrow==26.0.0", "huggingface-hub==1.33.0"
    )
    .env({"PYTHONPATH": "/project", "RAYON_NUM_THREADS": "2"})
    .add_local_file("tokenizer_research.py", "/project/tokenizer_research.py")
    .add_local_file("tokenizer_evaluation.py", "/project/tokenizer_evaluation.py")
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer", "/project/tokenizer")
)


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=7200,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume},
)
def collect_evaluation(run_id):
    from tokenizer_evaluation import EvaluationBuilder, validation_rows

    training = Path("/data/tokenizer-research") / run_id
    root = training / "balanced-evaluation-v2"
    builder = EvaluationBuilder(root, training)
    last_commit = time.monotonic()

    def checkpoint():
        nonlocal last_commit
        if time.monotonic() - last_commit >= 60:
            data_volume.commit()
            last_commit = time.monotonic()

    try:
        if (root / "frozen/manifest.json").exists():
            return json.loads((root / "frozen/manifest.json").read_text())
        for _ in range(20):
            for source in builder.recipe["sources"]:
                name = source["name"]
                if not builder.complete(name):
                    print(f"Collecting broader evaluation: {name}", flush=True)
                    builder.collect(name, reader=validation_rows, checkpoint=checkpoint)
                    data_volume.commit()
            if all(builder.complete(s["name"]) for s in builder.recipe["sources"]):
                result = builder.freeze()
                data_volume.commit()
                return result
        raise ValueError("Evaluation quotas incomplete")
    finally:
        builder.close()
        data_volume.commit()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=1200,
    retries=0,
    max_containers=3,
    volumes={"/data": data_volume, "/results": results_volume},
)
def evaluate_candidate(run_id, size, vocab):
    from tokenizer_evaluation import rescore

    data_volume.reload()
    folder = Path("/results/tokenizer-research") / run_id / f"{size}mib-{vocab}"
    result = rescore(
        folder, Path("/data/tokenizer-research") / run_id / "balanced-evaluation-v2/frozen"
    )
    results_volume.commit()
    print(f"Rescored {size} MiB / {vocab}: {result['metrics']['bytes_per_token']:.4f}", flush=True)
    return result


@app.local_entrypoint()
def main(run_id: str, phase: str = "collect"):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id) or phase not in ("collect", "evaluate"):
        raise ValueError("Invalid run ID or phase")
    out = Path("results/tokenizer-learning-curve") / run_id
    out.mkdir(parents=True, exist_ok=True)
    if phase == "collect":
        result = collect_evaluation.remote(run_id)
        (out / "evaluation-corpus-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        spec = json.loads(Path("tokenizer/research/learning-curve-v1.json").read_text())
        staged = Path("runs") / run_id / "balanced-results"
        staged.mkdir(parents=True, exist_ok=True)
        matrix = [(run_id, s, v) for s in spec["sizes_mib"] for v in spec["vocab_sizes"]]
        for result in evaluate_candidate.starmap(matrix, order_outputs=False):
            name = f"{result['size_mib']}mib-{result['requested_vocab_size']}.json"
            (staged / name).write_text(json.dumps(result, indent=2) + "\n")
        # Publish the comparable set only after every rescore succeeds.
        for path in staged.glob("*mib-*.json"):
            (out / path.name).write_bytes(path.read_bytes())
