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
    single_use_containers=True,
    volumes={"/runs": run_volume},
)
def worker(run_id, examples, settings, parent=PARENT, resume="", stop_after_step=None):
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
            stop_after_step=stop_after_step,
        )
    finally:
        run_volume.commit()


@app.function(image=image, cpu=2, memory=12288, timeout=300, volumes={"/runs": run_volume})
def compare(left, right):
    import torch

    run_volume.reload()
    a = torch.load(checkpoint_path(left), map_location="cpu", weights_only=False)
    b = torch.load(checkpoint_path(right), map_location="cpu", weights_only=False)

    def equal(x, y):
        if isinstance(x, torch.Tensor):
            return isinstance(y, torch.Tensor) and torch.equal(x, y)
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
        if isinstance(x, (tuple, list)):
            return len(x) == len(y) and all(equal(v, w) for v, w in zip(x, y))
        return x == y

    checks = {
        key: equal(a[key], b[key])
        for key in (
            "identity",
            "model",
            "optimizer",
            "step",
            "answer_tokens",
            "sampling_rng",
            "torch_rng",
            "cuda_rng",
        )
    }
    if not all(checks.values()):
        raise ValueError(f"Uninterrupted/resume comparison failed: {checks}")
    return {"passed": True, "exact": checks, "step": a["step"]}


@app.local_entrypoint()
def qualify(dataset: str, run_prefix: str):
    from trainer.repair_sft import Settings

    examples = Path(dataset).read_bytes()
    settings = Settings(steps=2, batch_size=1, evaluate_every=1, save_every=1).__dict__
    full = worker.remote(run_prefix + "-full", examples, settings)
    partial = worker.remote(run_prefix + "-part", examples, settings, stop_after_step=1)
    resumed = worker.remote(
        run_prefix + "-resume",
        examples,
        settings,
        resume=str(Path(partial["checkpoint"]).relative_to("/runs")),
    )
    report = compare.remote(
        str(Path(full["checkpoint"]).relative_to("/runs")),
        str(Path(resumed["checkpoint"]).relative_to("/runs")),
    )
    report.update(full=full, partial=partial, resumed=resumed)
    Path(dataset).with_suffix(".qualification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


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
