"""Resolve a selectable workflow, model configuration and dataset configuration."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from training_data.growing import max_persist_seconds

from distributed.config import DistributedConfig

from .checks import validate_paths


def relative_json(value):
    if not isinstance(value, str):
        raise ValueError("Expected a relative JSON path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
        raise ValueError("Expected a relative JSON path without parent traversal")
    return path


@dataclass(frozen=True)
class WorkflowConfig:
    model_config: str
    dataset_config: str
    gpu: str
    world_size: int
    target_tokens: int
    segment_seconds: int
    worker_timeout_seconds: int
    max_segments: int
    segments_per_call: int
    qualification_segments: int = 0
    qualification_steps: int = 2
    pilot: bool = False
    allow_unreviewed_data: bool = False
    source_workers: int = 3
    tokenizer_threads: int = 8
    database_batch_size: int = 64
    preparation_chunk_seconds: int = 20
    # How often preparation copies its staging database to the volume, in the background.
    persist_seconds: int = 600
    # Start training on a growing snapshot once every source has a usable prefix.
    overlap_preparation: bool = False

    def __post_init__(self):
        relative_json(self.model_config)
        relative_json(self.dataset_config)
        for name in ("pilot", "allow_unreviewed_data", "overlap_preparation"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"Workflow {name} must be a boolean")
        for name in (
            "world_size",
            "target_tokens",
            "segment_seconds",
            "worker_timeout_seconds",
            "max_segments",
            "segments_per_call",
            "qualification_steps",
            "source_workers",
            "tokenizer_threads",
            "database_batch_size",
            "preparation_chunk_seconds",
            "persist_seconds",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Workflow {name} must be a positive integer")
        if self.gpu not in ("H100", "L40S", "A10G") or self.world_size not in (1, 2, 3, 8):
            raise ValueError("Unsupported workflow GPU allocation")
        if not self.segment_seconds <= 82800 or not (
            self.segment_seconds + 480 < self.worker_timeout_seconds <= 86400
        ):
            raise ValueError("Workflow must leave time for process shutdown and checkpointing")
        if self.segments_per_call * self.worker_timeout_seconds >= 86400:
            raise ValueError("Workflow segments must fit within the controller timeout")
        # Training that caught up waits for the next persisted copy; it fails after
        # STALL_SECONDS, which must also cover the copy and the refresher's interval.
        if self.overlap_preparation and self.persist_seconds > max_persist_seconds():
            raise ValueError("Overlapped preparation must persist within half the stall timeout")
        if type(self.qualification_segments) is not int or not (
            0 <= self.qualification_segments < self.max_segments
        ):
            raise ValueError("Workflow must leave a segment after qualification")


@dataclass(frozen=True)
class DatasetConfig:
    mode: str
    root: str
    loader: str
    recipe: str = ""
    snapshot: str = ""

    def __post_init__(self):
        if not isinstance(self.root, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", self.root):
            raise ValueError("Dataset root must name one directory on the data volume")
        if self.loader not in ("raw", "token_shards", "live"):
            raise ValueError("Unsupported dataset loader")
        if self.mode == "prepare_raw":
            relative_json(self.recipe)
            if self.loader != "raw" or self.snapshot:
                raise ValueError("Raw preparation requires the raw loader and no snapshot")
        elif self.mode == "snapshot":
            validate_paths("validation", self.snapshot, "")
            if self.recipe:
                raise ValueError("Prepared datasets select a snapshot, not a preparation recipe")
        else:
            raise ValueError("Dataset mode must be prepare_raw or snapshot")


def resolve_settings(workflow, dataset, model):
    """Compose one trainer request with a single owner for each setting."""
    reserved = {"world_size", "target_tokens", "pilot", "allow_unreviewed_data", "data_loader"}
    if reserved.intersection(model):
        raise ValueError("Model configs cannot duplicate workflow or dataset settings")
    return DistributedConfig(
        **model,
        world_size=workflow.world_size,
        target_tokens=workflow.target_tokens,
        pilot=workflow.pilot,
        allow_unreviewed_data=workflow.allow_unreviewed_data,
        data_loader=dataset.loader,
    )


def resolve_payload(payload):
    """Revalidate the frozen selection on every remote controller/worker boundary."""
    workflow = WorkflowConfig(**payload["workflow"])
    dataset = DatasetConfig(**payload["dataset"])
    training = resolve_settings(workflow, dataset, payload["model"])
    return workflow, dataset, training


def load_workflow_config(path):
    """Load an explicit workflow path; there is no implicit experiment selection."""
    return WorkflowConfig(**json.loads(Path(path).read_text()))


def load_workflow(path, *, model_config="", dataset_config="", project_root=None):
    """Resolve paths locally and freeze exact selected config contents for remote calls."""
    path = Path(path).resolve()
    workflow = load_workflow_config(path)
    overrides = {}
    if model_config:
        overrides["model_config"] = model_config
    if dataset_config:
        overrides["dataset_config"] = dataset_config
    workflow = WorkflowConfig(**{**asdict(workflow), **overrides})
    inputs = {"workflow": path}
    for key in ("model_config", "dataset_config"):
        selected = (path.parent / relative_json(getattr(workflow, key))).resolve()
        if not selected.is_relative_to(path.parent):
            raise ValueError("Selected config escapes the workflow directory")
        inputs[key] = selected
    payload = {
        "workflow": asdict(workflow),
        "model": json.loads(inputs["model_config"].read_text()),
        "dataset": json.loads(inputs["dataset_config"].read_text()),
        "config_sources": {
            key: {"path": str(value), "sha256": hashlib.sha256(value.read_bytes()).hexdigest()}
            for key, value in inputs.items()
        },
    }
    _, dataset, _ = resolve_payload(payload)
    if dataset.mode == "prepare_raw":
        from training_data.simple_llm import load_recipe

        project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
        recipe_path = (project_root / relative_json(dataset.recipe)).resolve()
        if not recipe_path.is_relative_to(project_root):
            raise ValueError("Preparation recipe escapes the project directory")
        payload["preparation_recipe"] = load_recipe(recipe_path)
        payload["config_sources"]["preparation_recipe"] = {
            "path": dataset.recipe,
            "sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        }
    return payload
