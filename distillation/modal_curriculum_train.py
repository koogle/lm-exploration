"""Qualify both fresh student sizes on a completed synthetic corpus pilot."""

import json
import re
from pathlib import Path

import modal

from distillation.modal_app import artifacts, image

app = modal.App("lm-curriculum-students", include_source=False)


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    timeout=7200,
    retries=0,
    volumes={"/distill": artifacts},
    max_containers=1,
)
def train(corpus_id, run_id):
    from distillation.curriculum_train import run

    if not all(re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", x) for x in (corpus_id, run_id)):
        raise ValueError("Invalid run IDs")
    corpus = Path("/distill") / corpus_id
    if json.loads((corpus / "report.json").read_text())["status"] != "completed":
        raise ValueError("Corpus generation must be complete")
    try:
        return run(corpus, Path("/distill") / run_id, artifacts.commit)
    finally:
        artifacts.commit()


@app.local_entrypoint()
def main(corpus_id: str, run_id: str):
    result = train.remote(corpus_id, run_id)
    Path(f"distillation/results/{run_id}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
