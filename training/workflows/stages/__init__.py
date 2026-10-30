"""Post-training stage files: one validator for every stage's workflow file.

A stage file names one run of one post-training stage:

    {
      "stage": "sft",
      "name": "...", "description": "...", "provenance": "...",
      "run_id": "<fresh run ID>",
      "parent": "<run>/step-NNNNNNNNN.pt",
      ...the stage's own inputs...,
      "settings": {...}
    }

`stage`, `run_id`, `parent` (a checkpoint of any stage, `trainer/checkpoints.py`) and
`settings` are common to every stage; `name`, `description` and `provenance` are optional.
Each stage module in `STAGES` declares only what is its own: `FIELDS` (its inputs, such as
SFT's `data`), `inputs(spec)` (those inputs, checked and with defaults) and
`settings(values)` (its trainer's settings with every default filled in). Trainers know
nothing about workflow files.
"""

import json
from pathlib import Path

from workflows.checks import checkpoint, run_id
from workflows.stages import sft

STAGES = {"sft": sft}
COMMON = {"stage", "name", "description", "provenance", "run_id", "parent", "settings"}


def validate(spec):
    """A normalized copy of a stage file's contents, or ValueError."""
    if not isinstance(spec, dict) or spec.get("stage") not in STAGES:
        raise ValueError(f"A stage file names its stage, one of {sorted(STAGES)}")
    stage = STAGES[spec["stage"]]
    unknown = set(spec) - COMMON - stage.FIELDS
    if unknown:
        raise ValueError(f"Unknown {spec['stage']} fields: {sorted(unknown)}")
    for field in ("run_id", "parent", "settings"):
        if not spec.get(field):
            raise ValueError(f"A stage file needs an explicit {field}")
    if not isinstance(spec["settings"], dict):
        raise ValueError("settings must be an object")
    return {
        **{k: spec[k] for k in ("name", "description", "provenance") if k in spec},
        "stage": spec["stage"],
        "run_id": run_id(spec["run_id"]),
        "parent": checkpoint(spec["parent"], "parent"),
        **stage.inputs(spec),
        "settings": stage.settings(spec["settings"]),
    }


def load(path, **overrides):
    """A stage file's validated contents; nonempty `overrides` (such as `run_id`) replace
    its fields."""
    spec = json.loads(Path(path).read_text())
    spec.update({k: v for k, v in overrides.items() if v})
    return validate(spec)
