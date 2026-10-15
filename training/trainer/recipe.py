"""Training recipes: which model, which dataset, which settings and which GPU.

A recipe is a JSON file in ../recipes/ that maps directly onto the Recipe
dataclass below. Unknown, missing or invalid values fail when the file is
loaded, before any Modal worker starts. No run starts on import.
"""

import json
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from model_10m import model as model_10m
from model_100m import model as model_100m
from training_data.fetching import fetch_shakespeare_complete, fetch_tinyshakespeare

from .config import TrainConfig, validate_train_args

MODELS = {"10m": model_10m, "100m": model_100m}
DATASETS = {
    "tiny-shakespeare": fetch_tinyshakespeare,
    "shakespeare-complete": fetch_shakespeare_complete,
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


@dataclass(frozen=True)
class Backend:
    """Modal worker resources."""

    gpu: str = "L4"
    cpu: int = 4
    memory_mib: int = 16384
    max_seconds: int = 1200  # Cooperative training limit; the trainer saves and stops.
    timeout_seconds: int = 1500  # Hard worker limit, also the budget reservation.

    def __post_init__(self):
        if not 0 < self.max_seconds < self.timeout_seconds:
            raise ValueError("backend needs 0 < max_seconds < timeout_seconds")


@dataclass(frozen=True)
class Recipe:
    model: str  # A key of MODELS.
    dataset: str  # A key of DATASETS.
    training: TrainConfig
    model_config: dict = field(default_factory=dict)  # ModelConfig overrides, e.g. n_layer.
    backend: Backend = Backend()
    description: str = ""
    smoke: bool = False

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}")
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {sorted(DATASETS)}")
        validate_train_args(self.training)
        # A placeholder vocabulary rejects unknown or invalid dimensions now, not on the GPU.
        self.build_model_config(vocab_size=1)

    def build_model_config(self, vocab_size):
        # vocab_size always comes from the dataset's characters, never from the recipe.
        return MODELS[self.model].ModelConfig(vocab_size=vocab_size, **self.model_config)

    def as_smoke(self):
        return replace(
            self,
            training=replace(self.training, **SMOKE_TRAINING),
            backend=replace(self.backend, **SMOKE_BACKEND),
            smoke=True,
        )


def load_recipe(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    training = TrainConfig(**data.pop("training", {}))
    backend = Backend(**data.pop("backend", {}))
    return Recipe(**data, training=training, backend=backend)


def select_best_checkpoint(run_dir):
    """Copy the saved checkpoint with the lowest validation loss to best.pt."""
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    saved = [
        row
        for row in metrics
        if "val_loss" in row and (run_dir / "checkpoints" / f"step-{row['step']:06d}.pt").is_file()
    ]
    if not saved:
        raise ValueError("no evaluated checkpoint available")
    best = min(saved, key=lambda row: row["val_loss"])
    shutil.copyfile(run_dir / "checkpoints" / f"step-{best['step']:06d}.pt", run_dir / "best.pt")
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
        "recipe": asdict(recipe),
        "gpu": torch.cuda.get_device_name(),
        "config": json.loads((run_dir / "config.json").read_text()),
        "selected": best,
        "final": final,
        "worker_elapsed_seconds": time.perf_counter() - started,
        "sample": (run_dir / "sample.txt").read_text(),
        "selection": "lowest measured validation loss among saved checkpoints",
        "limits": "one seed; contiguous 90/10 split; no sealed test or broad capability claims",
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
