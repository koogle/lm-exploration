"""Historical character-recipe validation and model registry for checkpoint loading.

These schemas preserve old recipe interpretation; the character training loop and
launcher have been removed. Active recipes use distributed.config.DistributedConfig.
"""

from pathlib import Path
from typing import Literal, get_args

from model_10m import model as model_10m
from model_100m import model as model_100m
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
from training_data.fetching import fetch_shakespeare_complete, fetch_tinyshakespeare

from .config import TrainConfig

ModelName = Literal["10m", "100m"]
DatasetName = Literal["tiny-shakespeare", "shakespeare-complete"]
GpuName = Literal["T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100", "H200", "B200"]

MODELS = {
    "10m": model_10m,
    "100m": model_100m,
}
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


def load_recipe(path):
    """Validate a historical character recipe locally; no training entrypoint."""
    from training_data.registry import load_registry

    recipe = Recipe.model_validate_json(Path(path).read_text(encoding="utf-8"))
    if not load_registry()[recipe.dataset].training_eligible:
        raise ValueError(f"dataset {recipe.dataset!r} is not training eligible")
    return recipe
