"""Authorized 50M shared-tokenizer run on one H100; no validation early stopping."""

import json
import re
from pathlib import Path

import modal

from distillation.modal_app import artifacts, image

app = modal.App("lm-distillation-shared-50m", include_source=False)


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={
        "/distill": artifacts,
        "/data": modal.Volume.from_name("lm-exploration-mixture-data"),
        "/runs": modal.Volume.from_name("lm-exploration-training"),
    },
)
def worker(run_id, target_tokens=50_000_000, resume=""):
    from distillation.shared_train import run

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    if not 1 <= target_tokens <= 50_000_000:
        raise ValueError("Maximum authorized target is 50M")
    if resume and not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}/step-[0-9]{9}\.pt", resume):
        raise ValueError("Invalid resume checkpoint")
    try:
        return run(
            "/distill/" + run_id,
            artifacts.commit,
            target_tokens=target_tokens,
            resume=resume,
        )
    finally:
        artifacts.commit()


@app.local_entrypoint()
def main(run_id: str, target_tokens: int = 50_000_000, resume: str = ""):
    report = worker.remote(run_id, target_tokens, resume)
    Path("distillation/artifacts").mkdir(exist_ok=True)
    Path("distillation/artifacts", run_id + ".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
