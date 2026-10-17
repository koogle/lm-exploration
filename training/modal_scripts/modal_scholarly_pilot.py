"""Bounded qualification/pilot for the merged 124M subword model on Modal.

Run from training/. Pending corpus audits are allowed ONLY for this small pilot.
The unchanged modern architecture uses explicit causal attention, not FlashAttention.
"""

import json
import re
import subprocess
from pathlib import Path

import modal

app = modal.App("lm-exploration-scholarly-pilot")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
run_volume = modal.Volume.from_name("lm-exploration-training")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "numpy", "pydantic>=2.11,<3", "tokenizers==0.22.2")
    .env({"PYTHONPATH": "/project"})
    .add_local_dir("100m", "/project/model_100m")
    .add_local_dir("10m", "/project/model_10m")
    .add_local_dir("trainer", "/project/trainer")
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer", "/project/tokenizer")
    .add_local_dir("tests", "/project/tests")
    .add_local_file("generate.py", "/project/generate.py")
    .add_local_dir("modal_scripts", "/project/modal_scripts", ignore=["__pycache__"])
)


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def pilot(run_id: str, snapshot: str, provenance: dict):
    from trainer.subword_pilot import train_pilot

    try:
        report = train_pilot(
            Path("/data/scholarly-code") / snapshot / "manifest.json",
            Path("/runs") / run_id,
            provenance,
            run_volume.commit,
        )
        # Keep the launcher lightweight: return only JSON primitives, never framework objects.
        return json.loads(json.dumps(report))
    finally:
        run_volume.commit()


@app.function(image=image, cpu=2, memory=4096, timeout=300, retries=0, max_containers=1)
def qualify():
    for filename in ("test_100m_modern.py", "test_tied_embeddings.py", "test_scholarly.py"):
        subprocess.run(
            ["python", "-m", "unittest", "discover", "-s", "/project/tests", "-p", filename, "-v"],
            check=True,
            timeout=120,
        )
    return {"qualification": "passed"}


@app.local_entrypoint()
def main(run_id: str = "", snapshot: str = "", qualify_only: bool = False):
    if qualify_only:
        print(qualify.remote())
        return
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/snapshots/[a-zA-Z0-9_-]+", snapshot):
        raise ValueError("Expected version/snapshots/name")
    import hashlib

    files = [
        Path("modal_scripts/modal_scholarly_pilot.py"),
        *Path("trainer").glob("*.py"),
        *Path("data").glob("*.py"),
        Path("100m/model.py"),
        Path("10m/model.py"),
        Path("10m/legacy.py"),
    ]
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "purpose": "bounded pilot; corpus audits pending; not 20B pretraining",
    }
    qualify.remote()
    report = pilot.remote(run_id, snapshot, provenance)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{run_id}.json").write_text(json.dumps(report, indent=2) + "\n")
