"""The dataset registry in datasets.json: source, license review and eligibility."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

REGISTRY = Path(__file__).resolve().parent / "datasets.json"


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str
    name: str
    path: str  # Relative to data/.
    description: str
    source: str
    license_notes: str
    languages: list[str] = Field(min_length=1)
    training_eligible: bool


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    datasets: list[DatasetEntry]


def load_registry():
    registry = Registry.model_validate_json(REGISTRY.read_text(encoding="utf-8"))
    return {entry.id: entry for entry in registry.datasets}
