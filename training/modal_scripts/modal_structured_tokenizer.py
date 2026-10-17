"""Run bounded structured-output tokenizer comparisons on Modal."""

import json
from pathlib import Path

import modal

app = modal.App("lm-exploration-structured-tokenizer")
data = modal.Volume.from_name("lm-exploration-mixture-data")
results = modal.Volume.from_name("lm-exploration-training")
BASE = "learning-curve-20261016-v2"
RUN = "structured-code-v1"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "tokenizers==0.22.2", "requests==2.34.2", "pyarrow==26.0.0", "huggingface-hub==1.33.0"
    )
    .env({"PYTHONPATH": "/project", "RAYON_NUM_THREADS": "4"})
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer", "/project/tokenizer")
    .add_local_file(
        "tests/test_structured_tokenizer.py", "/project/tests/test_structured_tokenizer.py"
    )
)


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(8192, 8192),
    timeout=1800,
    max_containers=1,
    retries=0,
    volumes={"/data": data},
)
def prepare():
    import subprocess

    from tokenizer.structured_tokenizer import build

    subprocess.run(
        [
            "python",
            "-m",
            "unittest",
            "discover",
            "-s",
            "/project/tests",
            "-p",
            "test_structured_tokenizer.py",
            "-v",
        ],
        check=True,
    )
    root = Path("/data/tokenizer-research")
    out = root / RUN
    if out.exists():
        raise ValueError("Preserve previous run; choose a new version")
    result = build(
        root / BASE / "frozen/code_python-train.jsonl",
        root / BASE / "balanced-evaluation-v2/frozen/code_python-validation.jsonl",
        out,
    )
    data.commit()
    print("Prepared", result["actual_bytes"], flush=True)
    return result


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(8192, 8192),
    timeout=1800,
    max_containers=3,
    retries=0,
    volumes={"/data": data, "/results": results},
)
def fit_new(vocab):
    from tokenizers import Tokenizer

    from tokenizer.structured_tokenizer import fit
    from tokenizer.tokenizer_research import evaluate

    data.reload()
    out = Path("/results/tokenizer-research") / RUN / f"64mib-{vocab}"
    info = fit(Path("/data/tokenizer-research") / RUN, out, vocab)
    broad = Path("/data/tokenizer-research") / BASE / "balanced-evaluation-v2/frozen"
    manifest = json.loads((broad / "manifest.json").read_text())
    metrics = evaluate(Tokenizer.from_file(str(out / "tokenizer.json")), broad, manifest)
    info["general_metrics"] = {k: v for k, v in metrics.items() if k != "parent_totals"}
    info["general_evaluation_sha256"] = (
        __import__("hashlib").sha256((broad / "manifest.json").read_bytes()).hexdigest()
    )
    (out / "fit.json").write_text(json.dumps(info, indent=2) + "\n")
    results.commit()
    print("Fitted", vocab, flush=True)
    return info


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=1200,
    max_containers=3,
    retries=0,
    volumes={"/data": data, "/results": results},
)
def evaluate_candidate(family, size, vocab):
    from tokenizer.structured_tokenizer import score

    data.reload()
    results.reload()
    run = BASE if family == "original" else RUN
    candidate = Path("/results/tokenizer-research") / run / f"{size}mib-{vocab}/tokenizer.json"
    result = score(
        candidate,
        Path("/data/tokenizer-research") / RUN / "evaluation.jsonl",
        f"{family}-{size}mib-{vocab}",
    )
    result.update(family=family, size_mib=size)
    out = Path("/results/tokenizer-research") / RUN / "scores"
    out.mkdir(exist_ok=True)
    (out / (result["name"] + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    results.commit()
    print("Scored", result["name"], flush=True)
    return result


@app.local_entrypoint()
def main():
    out = Path("results/tokenizer-structured") / RUN
    out.mkdir(parents=True, exist_ok=True)
    manifest = prepare.remote()
    (out / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    fitted = list(fit_new.map([16384, 32768, 50000]))
    (out / "focused-fits.json").write_text(json.dumps(fitted, indent=2) + "\n")
    grid = [
        ("original", s, v) for s in [64, 128, 256, 512, 1024, 2048] for v in [16384, 32768, 50000]
    ]
    grid += [("focused", 64, v) for v in [16384, 32768, 50000]]
    measured = list(evaluate_candidate.starmap(grid, order_outputs=False))
    raw = Path("runs") / RUN
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "scores.json").write_text(json.dumps(measured, indent=2) + "\n")
    (out / "execution.json").write_text(
        json.dumps(
            {
                "execution": "modal-cpu",
                "remote_tests_passed": 2,
                "candidates_scored": len(measured),
                "focused_fits": 3,
                "correctness_passed": all(r["correctness_passed"] for r in measured),
                "prepare": {"cpu": 4, "memory_mib": 8192, "timeout_seconds": 1800},
                "fit": {
                    "cpu": 4,
                    "memory_mib": 8192,
                    "timeout_seconds": 1800,
                    "max_concurrency": 3,
                },
                "score": {
                    "cpu": 2,
                    "memory_mib": 4096,
                    "timeout_seconds": 1200,
                    "max_concurrency": 3,
                },
                "resource_settings": "CPU/memory requests and hard caps",
            },
            indent=2,
        )
        + "\n"
    )
