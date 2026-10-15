"""Training recipes: one JSON file names an architecture, a registered dataset,
training settings and backend resources. No run starts on import.

load_recipe() resolves and validates a recipe locally, before any paid worker
starts; run_recipe() executes the resolved dict on a Modal GPU worker. See ../recipes/README.md for the file format.
"""

import importlib
import json
import re
import shutil
import time
from dataclasses import fields
from pathlib import Path

import training_data

from .config import TrainConfig, validate_train_args

RUN_ID = re.compile(r"[a-zA-Z0-9_-]{1,80}")
ARCHITECTURE = re.compile(r"model_[a-z0-9_]+")
DATA_DIR = Path(training_data.__file__).resolve().parent
# data is chosen by the runner; resume is not part of a fresh recipe run.
TRAINING_KEYS = {field.name for field in fields(TrainConfig)} - {"data", "resume"}
BACKEND_DEFAULTS = {
    "kind": "modal",
    "gpu": "L4",
    "cpu": 4,
    "memory_mib": 16384,
    "max_seconds": 1200,  # Cooperative training limit; the trainer saves and stops.
    "timeout_seconds": 1500,  # Hard worker limit, also the budget reservation.
}
# Every recipe has the same smoke variant: a few updates to prove image, data and saving.
SMOKE_TRAINING = {
    "steps": 5,
    "batch_size": 2,
    "eval_interval": 5,
    "log_interval": 5,
    "eval_batches": 2,
    "checkpoint_interval": 5,
}
SMOKE_BACKEND = {"max_seconds": 60, "timeout_seconds": 180}


def dataset_entry(dataset_id):
    manifest = json.loads((DATA_DIR / "datasets.json").read_text(encoding="utf-8"))
    entry = next((item for item in manifest["datasets"] if item["id"] == dataset_id), None)
    if entry is None:
        raise ValueError(f"unknown dataset {dataset_id!r}; see data/datasets.json")
    if entry["training_eligible"] is not True:
        raise ValueError(f"dataset {dataset_id!r} is not training eligible")
    if not isinstance(entry.get("fetch"), str) or not re.fullmatch(r"\w+", entry["fetch"]):
        raise ValueError(f"dataset {dataset_id!r} needs a fetch module to run remotely")
    return entry


def load_architecture(name):
    if not isinstance(name, str) or not ARCHITECTURE.fullmatch(name):
        raise ValueError("model.architecture must be an installed package such as model_10m")
    return importlib.import_module(f"{name}.model")


def model_config(architecture, overrides, vocab_size):
    if "vocab_size" in overrides:
        raise ValueError("vocab_size comes from the dataset tokenizer, not the recipe")
    return architecture.ModelConfig(vocab_size=vocab_size, **overrides)


def load_recipe(path, *, smoke=False):
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(recipe) - {"description", "model", "dataset", "training", "backend"}
    if unknown:
        raise ValueError(f"unknown recipe keys: {sorted(unknown)}")
    model = recipe.get("model")
    if not isinstance(model, dict) or set(model) - {"architecture", "config"}:
        raise ValueError("model must be {'architecture': ..., 'config': {...}}")
    overrides = model.get("config", {})
    # Construct once with a placeholder vocabulary to reject unknown or invalid dimensions.
    model_config(load_architecture(model.get("architecture")), overrides, vocab_size=1)
    training = recipe.get("training", {})
    if set(training) - TRAINING_KEYS:
        raise ValueError(f"unknown training keys: {sorted(set(training) - TRAINING_KEYS)}")
    backend = recipe.get("backend", {})
    if set(backend) - BACKEND_DEFAULTS.keys():
        raise ValueError(f"unknown backend keys: {sorted(set(backend) - BACKEND_DEFAULTS.keys())}")
    backend = BACKEND_DEFAULTS | backend
    if backend["kind"] != "modal":
        raise ValueError("only the modal backend is supported")
    if smoke:
        training = training | SMOKE_TRAINING
        backend = backend | SMOKE_BACKEND
    if not 0 < backend["max_seconds"] < backend["timeout_seconds"]:
        raise ValueError("backend needs 0 < max_seconds < timeout_seconds")
    validate_train_args(TrainConfig(data="", **training))
    return {
        "name": Path(path).stem,
        "description": recipe.get("description", ""),
        "smoke": smoke,
        "model": {"architecture": model["architecture"], "config": overrides},
        "dataset": dataset_entry(recipe.get("dataset")),
        "training": training,
        "backend": backend,
    }


def run_recipe(recipe, run_dir):
    """Fetch the dataset, train, select the best saved checkpoint and write report.json."""
    import torch
    from training_data.loading import load_data

    from .loop import train

    run_dir = Path(run_dir)
    if not RUN_ID.fullmatch(run_dir.name):
        raise ValueError("invalid run ID")
    entry = recipe["dataset"]
    importlib.import_module(f"training_data.fetching.{entry['fetch']}").main()
    data = DATA_DIR / entry["path"]
    architecture = load_architecture(recipe["model"]["architecture"])
    overrides = recipe["model"]["config"]
    block_size = model_config(architecture, overrides, vocab_size=1).block_size
    tokenizer, _, _, _ = load_data(data, block_size)
    args = TrainConfig(data=str(data), **recipe["training"])
    started = time.perf_counter()
    train(
        args,
        run_dir,
        architecture_module=architecture,
        model_config=model_config(architecture, overrides, len(tokenizer.characters)),
        max_seconds=recipe["backend"]["max_seconds"],
    )
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    candidates = [
        row
        for row in metrics
        if "val_loss" in row and (run_dir / "checkpoints" / f"step-{row['step']:06d}.pt").is_file()
    ]
    if not candidates:
        raise ValueError("no evaluated checkpoint available")
    best = min(candidates, key=lambda row: row["val_loss"])
    shutil.copyfile(run_dir / "checkpoints" / f"step-{best['step']:06d}.pt", run_dir / "best.pt")
    report = {
        "run_id": run_dir.name,
        "recipe": recipe,
        "smoke": recipe["smoke"],
        "gpu": torch.cuda.get_device_name(),
        "config": json.loads((run_dir / "config.json").read_text()),
        "selected": best,
        "final": metrics[-1],
        "worker_elapsed_seconds": time.perf_counter() - started,
        "sample": (run_dir / "sample.txt").read_text(),
        "selection": "lowest measured validation loss among saved checkpoints",
        "limits": "one seed; contiguous 90/10 split; no sealed test or broad capability claims",
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
