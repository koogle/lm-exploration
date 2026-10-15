"""Training settings. Runs come from recipes (../recipes/) and train on a Modal GPU."""

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

# Training runs only on Modal GPU workers; there is no CPU or MPS training path.
DEVICE = "cuda"


class TrainConfig(BaseModel):
    """Hyperparameters only; the dataset and any resume checkpoint are train() arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    steps: PositiveInt = 100
    batch_size: PositiveInt = 1
    lr: float = Field(3e-4, gt=0, allow_inf_nan=False)
    seed: int = Field(1337, ge=0, le=2**63 - 3)
    eval_interval: PositiveInt = 10
    log_interval: PositiveInt = 1
    eval_batches: PositiveInt = 2
    checkpoint_interval: PositiveInt = 10
