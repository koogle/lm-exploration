"""Historical character-recipe schema; active settings use distributed.config."""

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class TrainConfig(BaseModel):
    """Recorded character-training hyperparameters; not the active loop's settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    steps: PositiveInt = 100
    batch_size: PositiveInt = 1
    lr: float = Field(3e-4, gt=0, allow_inf_nan=False)
    seed: int = Field(1337, ge=0, le=2**63 - 3)
    eval_interval: PositiveInt = 10
    log_interval: PositiveInt = 1
    eval_batches: PositiveInt = 2
    checkpoint_interval: PositiveInt = 10
