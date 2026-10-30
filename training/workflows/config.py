"""Resolve one pretraining or mid-training stage: its execution settings, model
configuration and dataset configuration (`workflows/plan.py` resolves whole workflows).

A stage trains a model `family` (the workflow's):

- `subword`: `model_config` (relative to training/) holds the subword GPT's
  architecture and optimizer settings and the dataset config its data; the stage
  owns the token target, GPU count and data-review opt-in, and the three resolve
  into one `DistributedConfig`.
- `hybrid`: `model_config` (relative to training/) is a hybrid run config
  (hybrid/configs/*.json), which owns its data recipe and token target; the stage
  owns only execution (GPUs, sessions, qualification).
"""

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


FAMILIES = ("subword", "hybrid")


@dataclass(frozen=True)
class PretrainConfig:
    """The execution settings of one pretraining or mid-training stage."""

    family: str
    model_config: str
    gpu: str
    world_size: int
    segment_seconds: int
    worker_timeout_seconds: int
    max_segments: int
    segments_per_call: int
    dataset_config: str = ""
    target_tokens: int = 0
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
        if self.family not in FAMILIES:
            raise ValueError(f"Family must be one of {FAMILIES}")
        relative_json(self.model_config)
        for name in ("pilot", "allow_unreviewed_data", "overlap_preparation"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"Workflow {name} must be a boolean")
        if self.family == "subword":
            relative_json(self.dataset_config)
            if type(self.target_tokens) is not int or self.target_tokens < 1:
                raise ValueError("Workflow target_tokens must be a positive integer")
        elif self.dataset_config or self.target_tokens or self.pilot or self.allow_unreviewed_data:
            raise ValueError("A hybrid run config owns its data and token target")
        for name in (
            "world_size",
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


def resolve_hybrid(workflow, run):
    """A hybrid run config, checked against the workflow's GPU count (no torch import)."""
    required = ("recipe", "model", "target_tokens", "global_rows", "microbatch")
    if not isinstance(run, dict) or any(key not in run for key in required):
        raise ValueError(f"A hybrid run config needs {required}")
    rows, microbatch = run["global_rows"], run["microbatch"]
    if any(type(v) is not int or v < 1 for v in (rows, microbatch, run["target_tokens"])):
        raise ValueError("Hybrid rows, microbatch and target must be positive integers")
    if rows % (workflow.world_size * microbatch):
        raise ValueError("Global rows must divide into whole per-GPU microbatches")
    return run


def step_tokens(family, settings):
    """Tokens one optimizer step consumes at most (the family's token budget unit)."""
    if family == "hybrid":
        return settings["global_rows"] * settings["model"]["block_size"]
    return settings["global_windows"] * settings["context"]


def resolve_settings(workflow, dataset, model):
    """Compose one trainer request with a single owner for each setting."""
    if workflow.family == "hybrid":
        return resolve_hybrid(workflow, model)
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
    policy = PretrainConfig(**payload["stage"])
    dataset = DatasetConfig(**payload["dataset"]) if payload["dataset"] else None
    training = resolve_settings(policy, dataset, payload["model"])
    return policy, dataset, training


def project_file(project_root, value):
    """A JSON file named relative to training/, never outside it."""
    path = (project_root / relative_json(value)).resolve()
    if not path.is_relative_to(project_root):
        raise ValueError("A selected config escapes the project directory")
    return path


def pretrain_bundle(spec, family, dataset_config="", project_root=None):
    """Freeze a pretraining or mid-training stage for remote calls: its execution
    settings (`spec`'s other fields), and the contents and hashes of its model config,
    its dataset config (`dataset_config`, subword only) and preparation recipe."""
    project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    fields = {k: v for k, v in spec.items() if k not in ("stage", "name", "description", "data")}
    policy = PretrainConfig(family=family, dataset_config=dataset_config, **fields)
    inputs = {"model_config": project_file(project_root, policy.model_config)}
    if family == "subword":
        inputs["dataset_config"] = project_file(project_root, dataset_config)
    payload = {
        "stage": asdict(policy),
        "model": json.loads(inputs["model_config"].read_text()),
        "dataset": json.loads(inputs["dataset_config"].read_text())
        if "dataset_config" in inputs
        else None,
        "config_sources": {
            key: {"path": str(value), "sha256": hashlib.sha256(value.read_bytes()).hexdigest()}
            for key, value in inputs.items()
        },
    }
    _, dataset, _ = resolve_payload(payload)
    if dataset and dataset.mode == "prepare_raw":
        from training_data.simple_llm import load_recipe

        recipe_path = project_file(project_root, dataset.recipe)
        payload["preparation_recipe"] = load_recipe(recipe_path)
        payload["config_sources"]["preparation_recipe"] = {
            "path": dataset.recipe,
            "sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        }
    return payload
