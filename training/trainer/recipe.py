"""Training recipes: which model, which dataset, which settings and which GPU.

A recipe is a JSON file in ../recipes/ validated by the Recipe model below.
Unknown keys, wrong types and values outside the fixed choices fail when the
file is loaded, before any Modal worker starts. No run starts on import.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Literal, get_args

from model_10m import model as model_10m
from model_10m_rmsnorm import model as model_10m_rmsnorm
from model_100m import model as model_100m
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
from training_data.fetching import fetch_shakespeare_complete, fetch_tinyshakespeare

from .config import TrainConfig

ModelName = Literal["10m", "100m", "10m_rmsnorm"]
DatasetName = Literal["tiny-shakespeare", "shakespeare-complete"]
GpuName = Literal["T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100", "H200", "B200"]

MODELS = {"10m": model_10m, "100m": model_100m, "10m_rmsnorm": model_10m_rmsnorm}
DATASETS = {
    "tiny-shakespeare": fetch_tinyshakespeare,
    "shakespeare-complete": fetch_shakespeare_complete,
}
assert set(get_args(ModelName)) == MODELS.keys()
assert set(get_args(DatasetName)) == DATASETS.keys()

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


class Backend(BaseModel):
    """Modal worker resources."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    gpu: GpuName = "L4"
    cpu: PositiveInt = 4
    memory_mib: PositiveInt = 16384
    max_seconds: PositiveInt = 1200  # Cooperative training limit; the trainer saves and stops.
    timeout_seconds: PositiveInt = 1500  # Hard worker limit, also the budget reservation.

    @model_validator(mode="after")
    def check_limits(self):
        if self.max_seconds >= self.timeout_seconds:
            raise ValueError("max_seconds must be below timeout_seconds")
        return self


class Recipe(BaseModel):
    # protected_namespaces=() allows the model_overrides field name.
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, protected_namespaces=())

    model: ModelName
    dataset: DatasetName
    training: TrainConfig
    model_overrides: dict[str, int] = {}  # ModelConfig overrides, e.g. n_layer.
    backend: Backend = Backend()
    description: str = ""
    smoke: bool = False

    @model_validator(mode="after")
    def check_model_overrides(self):
        # A placeholder vocabulary rejects unknown or invalid dimensions now, not on the GPU.
        try:
            self.build_model_config(vocab_size=1)
        except TypeError as error:
            raise ValueError(str(error)) from error
        return self

    def build_model_config(self, vocab_size):
        # vocab_size always comes from the dataset's characters, never from the recipe.
        return MODELS[self.model].ModelConfig(vocab_size=vocab_size, **self.model_overrides)

    def as_smoke(self):
        return self.model_copy(
            update={
                "training": self.training.model_copy(update=SMOKE_TRAINING),
                "backend": self.backend.model_copy(update=SMOKE_BACKEND),
                "smoke": True,
            }
        )


class MetricRow(BaseModel):
    """One line of a run's metrics.jsonl, as written by trainer/loop.py."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int
    train_loss: float
    tokens_seen: int
    val_loss: float | None = None  # Only on evaluation steps.
    elapsed_seconds: float
    tokens_per_second: float


def load_recipe(path):
    # Local only: the registry is not part of the installed package on the worker.
    from training_data.registry import load_registry

    recipe = Recipe.model_validate_json(Path(path).read_text(encoding="utf-8"))
    if not load_registry()[recipe.dataset].training_eligible:
        raise ValueError(f"dataset {recipe.dataset!r} is not training eligible")
    return recipe


def select_best_checkpoint(run_dir):
    """Copy the saved checkpoint with the lowest validation loss to best.pt."""
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    metrics = [MetricRow.model_validate_json(line) for line in lines]
    saved = [
        row
        for row in metrics
        if row.val_loss is not None
        and (run_dir / "checkpoints" / f"step-{row.step:06d}.pt").is_file()
    ]
    if not saved:
        raise ValueError("no evaluated checkpoint available")
    best = min(saved, key=lambda row: row.val_loss)
    shutil.copyfile(run_dir / "checkpoints" / f"step-{best.step:06d}.pt", run_dir / "best.pt")
    return best, metrics[-1]


def run_recipe(recipe, run_dir):
    """Fetch the dataset, train, select the best checkpoint and write report.json."""
    import torch
    from training_data.loading import load_data

    from .loop import train

    run_dir = Path(run_dir)
    fetcher = DATASETS[recipe.dataset]
    fetcher.main()
    data = fetcher.DESTINATION
    tokenizer = load_data(data, recipe.build_model_config(vocab_size=1).block_size)[0]
    started = time.perf_counter()
    train(
        recipe.training,
        data,
        run_dir,
        architecture_module=MODELS[recipe.model],
        model_config=recipe.build_model_config(len(tokenizer.characters)),
        max_seconds=recipe.backend.max_seconds,
    )
    best, final = select_best_checkpoint(run_dir)
    report = {
        "run_id": run_dir.name,
        "recipe": recipe.model_dump(),
        "gpu": torch.cuda.get_device_name(),
        "config": json.loads((run_dir / "config.json").read_text()),
        "selected": best.model_dump(exclude_none=True),
        "final": final.model_dump(exclude_none=True),
        "worker_elapsed_seconds": time.perf_counter() - started,
        "sample": (run_dir / "sample.txt").read_text(),
        "selection": "lowest measured validation loss among saved checkpoints",
        "limits": "one seed; contiguous 90/10 split; no sealed test or broad capability claims",
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
