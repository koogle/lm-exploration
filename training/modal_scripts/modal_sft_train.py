"""Bounded SFT (math, repair, CIG) on one Modal GPU. Importing this module never starts a job.

Run from training/ (see sft/README.md for the recipe format):
    modal run -m modal_scripts.modal_sft_train::main --recipe <recipe.json> \
        [--run-id <fresh-id>] [--parent <run/step-N.pt>] [--dataset <jsonl> | --data-id <id>] \
        [--resume <run/step-N.pt>] [--output <result.json>]
    modal run -m modal_scripts.modal_sft_train::qualify --recipe <recipe.json> \
        --run-prefix <fresh-prefix> [--steps 2] [same --parent/--dataset/--data-id/--resume]

Each launch makes one training call (qualification: three sequential calls) with no retries
and a six-hour function timeout. The launcher records a deadline 24 hours after launch; a call
that starts after it refuses work, and training stops taking updates 30 minutes before it to
evaluate and save a final checkpoint. Checkpoints are saved on the private run volume.
"""

import json
import time
from pathlib import Path

import modal

from distributed.image import image

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
FINAL_RESERVE_SECONDS = 30 * 60
# Settings caps the training loop at four hours; reference scoring, evaluation and saving fit.
TIMEOUT_SECONDS = 6 * 60 * 60
LOCAL_DATA_LIMIT = 20 * 1024 * 1024

image = image.add_local_dir("modal_scripts", "/project/modal_scripts")
app = modal.App("simple-llm-sft")
run_volume = modal.Volume.from_name("lm-exploration-training")
limits = {
    "image": image,
    "timeout": TIMEOUT_SECONDS,
    "retries": 0,
    "max_containers": 2,
    "single_use_containers": True,
    "volumes": {"/runs": run_volume},
}


def run(recipe, examples, deadline, stop_after_step=None):
    from training_data.subword import digest

    from trainer.sft import Settings, train
    from trainer.sft_recipe import validate_recipe

    if time.time() >= deadline - FINAL_RESERVE_SECONDS:
        raise TimeoutError("Launch deadline passed; refusing SFT work")
    recipe = validate_recipe(recipe)
    run_volume.reload()
    runs = Path("/runs")
    if "volume" in recipe["data"]:
        folder = runs / recipe["data"]["volume"]
        report = json.loads((folder / "report.json").read_text())
        dataset = folder / "examples.jsonl"
        if digest(dataset) != report["dataset_sha256"]:
            raise ValueError("Dataset differs from its preparation report")
    else:
        if not examples or len(examples) > LOCAL_DATA_LIMIT:
            raise ValueError("Local SFT data must be nonempty and at most 20 MiB")
        dataset = Path("/tmp/sft-examples.jsonl")
        dataset.write_bytes(examples)
    try:
        result = train(
            runs / recipe["parent"],
            dataset,
            runs / recipe["run_id"],
            Settings(**recipe["settings"]),
            run_volume.commit,
            resume=runs / recipe["resume"] if recipe["resume"] else None,
            stop_after_step=stop_after_step,
            extend=recipe["extend"],
            deadline=deadline - FINAL_RESERVE_SECONDS,
        )
        (runs / recipe["run_id"] / "result.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        run_volume.commit()


@app.function(gpu="L40S", cpu=4, memory=16384, **limits)
def train_l40s(recipe, examples, deadline, stop_after_step=None):
    return run(recipe, examples, deadline, stop_after_step)


@app.function(gpu="H100", cpu=8, memory=65536, **limits)
def train_h100(recipe, examples, deadline, stop_after_step=None):
    return run(recipe, examples, deadline, stop_after_step)


WORKERS = {"L40S": train_l40s, "H100": train_h100}


@app.function(
    image=image, cpu=2, memory=12288, timeout=600, retries=0, volumes={"/runs": run_volume}
)
def compare(left, right, deadline):
    """Exact equality of two SFT checkpoints (uninterrupted versus restored)."""
    import torch

    from trainer.sft_recipe import checkpoint

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing comparison")
    run_volume.reload()
    a = torch.load(Path("/runs") / checkpoint(left), map_location="cpu", weights_only=False)
    b = torch.load(Path("/runs") / checkpoint(right), map_location="cpu", weights_only=False)

    def equal(x, y):
        if isinstance(x, torch.Tensor):
            return isinstance(y, torch.Tensor) and torch.equal(x, y)
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
        if isinstance(x, (tuple, list)):
            return len(x) == len(y) and all(equal(v, w) for v, w in zip(x, y, strict=True))
        return x == y

    keys = (
        "identity",
        "model",
        "optimizer",
        "step",
        "answer_tokens",
        "sampling_rng",
        "torch_rng",
        "cuda_rng",
    )
    checks = {key: equal(a[key], b[key]) for key in keys}
    return {"passed": all(checks.values()), "exact": checks, "step": a["step"]}


def local_examples(recipe):
    """Validate local data before allocating a GPU; volume data is checked remotely."""
    if "volume" in recipe["data"]:
        return b""
    from training_data.subword import load_frozen

    from trainer.sft import TASKS

    data = TASKS[recipe["settings"]["task"]]
    path = Path(recipe["data"]["local"])
    data.prepare(path, load_frozen(), recipe["settings"]["context"])
    return path.read_bytes()


def relative(path):
    return str(Path(path).relative_to("/runs"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"output": str(path)}))


@app.local_entrypoint()
def main(
    recipe: str,
    run_id: str = "",
    parent: str = "",
    dataset: str = "",
    data_id: str = "",
    resume: str = "",
    output: str = "",
):
    from trainer.sft_recipe import load_recipe

    spec = load_recipe(recipe, run_id, parent, dataset, data_id, resume)
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    examples = local_examples(spec)
    result = WORKERS[spec["gpu"]].remote(spec, examples, deadline)
    write(output or f"artifacts/sft/{spec['run_id']}.json", {"recipe": spec, "result": result})


@app.local_entrypoint()
def qualify(
    recipe: str,
    run_prefix: str,
    steps: int = 2,
    parent: str = "",
    dataset: str = "",
    data_id: str = "",
    resume: str = "",
    output: str = "",
):
    """Compare an uninterrupted run with stop-at-(steps-1), fresh-container restore, continue.

    A recipe with `resume` and `extend` qualifies an extension from that checkpoint; set
    `--steps` to its restored step plus two.
    """
    from trainer.sft_recipe import load_recipe, run_id

    spec = load_recipe(recipe, "", parent, dataset, data_id, resume)
    run_id(run_prefix + "-resume", "run_prefix")
    if steps < 2:
        raise ValueError("Qualification needs at least two steps")
    spec["settings"]["steps"] = steps
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    examples = local_examples(spec)
    worker = WORKERS[spec["gpu"]]
    full = worker.remote({**spec, "run_id": run_prefix + "-full"}, examples, deadline)
    part = worker.remote(
        {**spec, "run_id": run_prefix + "-part"}, examples, deadline, stop_after_step=steps - 1
    )
    resumed = worker.remote(
        {
            **spec,
            "run_id": run_prefix + "-resume",
            "resume": relative(part["checkpoint"]),
            "extend": False,
        },
        examples,
        deadline,
    )
    report = compare.remote(relative(full["checkpoint"]), relative(resumed["checkpoint"]), deadline)
    report.update(recipe=spec, full=full, partial=part, resumed=resumed)
    write(output or f"artifacts/sft/{run_prefix}-qualification.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Resume qualification failed: {report['exact']}")
