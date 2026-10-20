"""Explicit bounded repair SFT launch; importing this module never starts a job."""

import json
import re
from pathlib import Path

import modal

from modal_scholarly_pilot import image

app = modal.App("simple-llm-repair-sft")
run_volume = modal.Volume.from_name("lm-exploration-training")
PARENT = "simple-llm-code60-fresh-5b-20261019-r02-s02/step-000066065.pt"


def checkpoint_path(value):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}/step-[0-9]{9}\.pt", value):
        raise ValueError("Expected run-id/step-NNNNNNNNN.pt")
    return Path("/runs") / value


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/runs": run_volume},
)
def worker(run_id, examples, settings, parent=PARENT, resume=""):
    from trainer.repair_sft import Settings, train

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    if len(examples) > 20 * 1024 * 1024:
        raise ValueError("Pilot input limited to 20 MiB")
    parent_path = checkpoint_path(parent)
    resume_path = checkpoint_path(resume) if resume else None
    run_volume.reload()
    dataset = Path("/tmp/repair-examples.jsonl")
    dataset.write_bytes(examples)
    try:
        return train(
            parent_path,
            dataset,
            Path("/runs") / run_id,
            Settings(**settings),
            run_volume.commit,
            resume=resume_path,
        )
    finally:
        run_volume.commit()


@app.local_entrypoint()
def main(
    dataset: str,
    run_id: str,
    steps: int = 200,
    context: int = 1024,
    batch_size: int = 4,
    lr: float = 1e-5,
    max_seconds: int = 1200,
    parent: str = PARENT,
    resume: str = "",
):
    from training_data.repair_sft import prepare
    from training_data.subword import load_frozen

    from trainer.repair_sft import Settings

    settings = Settings(
        steps=steps, context=context, batch_size=batch_size, lr=lr, max_seconds=max_seconds
    )
    prepare(dataset, load_frozen(), context)  # Reject invalid inputs before allocating a GPU.
    checkpoint_path(parent)
    if resume:
        checkpoint_path(resume)
    print(
        json.dumps(
            worker.remote(run_id, Path(dataset).read_bytes(), settings.__dict__, parent, resume),
            indent=2,
        )
    )
