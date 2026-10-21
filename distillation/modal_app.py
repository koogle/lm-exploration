"""Run from repository root: modal run --detach -m distillation.modal_app."""

import json
import re
from pathlib import Path

import modal

app = modal.App("lm-exploration-distillation", include_source=False)
artifacts = modal.Volume.from_name(
    "lm-exploration-distillation", create_if_missing=True
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0", "transformers==4.57.1", "tokenizers==0.22.2", "numpy==2.2.6"
    )
    .env({"PYTHONPATH": "/project", "HF_HOME": "/distill/hf-cache"})
    .add_local_dir(
        "distillation", "/project/distillation", ignore=["results", "__pycache__"]
    )
    .add_local_dir("training/100m", "/project/model_100m")
    .add_local_dir("training/10m", "/project/model_10m")
    .add_local_dir(
        "training/data", "/project/training_data", ignore=["eu", "__pycache__"]
    )
    .add_local_dir(
        "training/tokenizer/structured-code-v1", "/project/tokenizer/structured-code-v1"
    )
)


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={
        "/distill": artifacts,
        "/data": modal.Volume.from_name("lm-exploration-mixture-data"),
        "/runs": modal.Volume.from_name("lm-exploration-training"),
    },
)
def worker(run_id, target_tokens=262144, max_seconds=1200, resume=""):
    from distillation.train import run

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    if not 1 <= target_tokens <= 5_000_000 or not 30 <= max_seconds <= 1200:
        raise ValueError(
            "This qualification launcher is bounded to 5M tokens / 20 minutes"
        )
    if resume and not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}/step-[0-9]{9}\.pt", resume):
        raise ValueError("Resume must name a checkpoint on the distillation volume")
    try:
        return run(
            "/distill/" + run_id, artifacts.commit, target_tokens, max_seconds, resume
        )
    finally:
        artifacts.commit()


@app.local_entrypoint()
def main(
    run_id: str, target_tokens: int = 262144, max_seconds: int = 1200, resume: str = ""
):
    report = worker.remote(run_id, target_tokens, max_seconds, resume)
    path = Path("distillation/results") / (run_id + ".json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
