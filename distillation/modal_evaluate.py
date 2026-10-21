"""Evaluate parent and pilot checkpoints without further optimization."""

import json
import re
from pathlib import Path

import modal

from distillation.modal_app import artifacts, image

app = modal.App("lm-distillation-evaluate", include_source=False)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=900,
    retries=0,
    volumes={
        "/distill": artifacts,
        "/data": modal.Volume.from_name("lm-exploration-mixture-data"),
        "/runs": modal.Volume.from_name("lm-exploration-training"),
    },
)
def worker(run_id):
    from distillation.evaluate import compare
    from distillation.train import PARENT

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    output = Path("/distill") / run_id
    output.mkdir(exist_ok=False)
    checkpoints = [
        "/runs/" + PARENT,
        "/distill/qwen4b-byte-pilot-20261021-02/step-000000500.pt",
        "/distill/qwen4b-byte-pilot-20261021-02/step-000001000.pt",
        "/distill/qwen4b-byte-pilot-20261021-02/step-000001133.pt",
    ]
    report = compare(checkpoints)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    artifacts.commit()
    return report


@app.local_entrypoint()
def main(run_id: str):
    report = worker.remote(run_id)
    Path("distillation/artifacts").mkdir(exist_ok=True)
    Path("distillation/artifacts", run_id + ".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            [
                {k: v for k, v in r.items() if k != "documents"}
                for r in report["results"]
            ],
            indent=2,
        )
    )
