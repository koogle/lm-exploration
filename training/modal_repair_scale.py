"""Volume-backed, bounded larger SFT. Deploying does not start training."""

import json
import re
from pathlib import Path

import modal

from modal_scholarly_pilot import image

image = image.add_local_file("modal_scholarly_pilot.py", "/project/modal_scholarly_pilot.py")
app = modal.App("simple-llm-repair-sft-scale")
volume = modal.Volume.from_name("lm-exploration-training")
PARENT = "simple-llm-code60-fresh-5b-20261019-r02-s02/step-000066065.pt"


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=18000,
    retries=0,
    max_containers=1,
    single_use_containers=True,
    volumes={"/runs": volume},
)
def worker(run_id, data_id, stop_after_step=None, resume="", run_settings=None):
    from training_data.subword import digest

    from trainer.repair_sft import Settings, train

    for name in (run_id, data_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", name):
            raise ValueError("Invalid run/data ID")
    if resume and not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}/step-[0-9]{9}\.pt", resume):
        raise ValueError("Invalid resume path")
    volume.reload()
    folder = Path("/runs") / data_id
    report = json.loads((folder / "report.json").read_text())
    if digest(folder / "examples.jsonl") != report["dataset_sha256"]:
        raise ValueError("Dataset differs from preparation report")
    defaults = {
        "steps": 5000,
        "batch_size": 8,
        "context": 2048,
        "lr": 1e-5,
        "evaluate_every": 250,
        "save_every": 250,
        "max_seconds": 14400,
    }
    if run_settings is not None:
        defaults.update(run_settings)
    settings = Settings(**defaults)
    try:
        result = train(
            Path("/runs") / PARENT,
            folder / "examples.jsonl",
            Path("/runs") / run_id,
            settings,
            volume.commit,
            resume=Path("/runs") / resume if resume else None,
            stop_after_step=stop_after_step,
        )
        (Path("/runs") / run_id / "result.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        volume.commit()
