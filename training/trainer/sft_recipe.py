"""Validate SFT run recipes for `modal_scripts/modal_sft_train.py` without Modal.

A recipe is a JSON object naming one run: the parent checkpoint, the data, the output run ID,
the GPU type and `trainer.sft.Settings`. Command-line overrides replace recipe values so
one recipe can describe a recorded run and also seed a fresh one.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path

from trainer.sft import Settings

RUN_ID = re.compile(r"[a-zA-Z0-9_-]{1,80}")
CHECKPOINT = re.compile(r"[a-zA-Z0-9_-]{1,80}/step-[0-9]{9}\.pt")
GPUS = ("L40S", "H100")
FIELDS = {
    "name",
    "description",
    "provenance",
    "run_id",
    "parent",
    "data",
    "resume",
    "extend",
    "gpu",
    "settings",
}


def run_id(value, field="run_id"):
    if not isinstance(value, str) or not RUN_ID.fullmatch(value):
        raise ValueError(f"{field} must match {RUN_ID.pattern}")
    return value


def checkpoint(value, field="checkpoint"):
    """Volume-relative `run-id/step-NNNNNNNNN.pt`; never an absolute or parent path."""
    if not isinstance(value, str) or not CHECKPOINT.fullmatch(value):
        raise ValueError(f"{field} must be run-id/step-NNNNNNNNN.pt")
    return value


def validate_recipe(recipe):
    """Return a normalized copy, or raise ValueError. Safe to call locally and remotely."""
    if not isinstance(recipe, dict):
        raise ValueError("Recipe must be a JSON object")
    unknown = set(recipe) - FIELDS
    if unknown:
        raise ValueError(f"Unknown recipe fields: {sorted(unknown)}")
    for field in ("run_id", "parent", "data", "settings"):
        if not recipe.get(field):
            raise ValueError(f"Recipe needs an explicit {field}")
    data = recipe["data"]
    if not isinstance(data, dict) or len(data) != 1 or not set(data) <= {"local", "volume"}:
        raise ValueError('data must be {"local": path} or {"volume": data-id}')
    if "volume" in data:
        run_id(data["volume"], "data.volume")
    elif not isinstance(data["local"], str) or not data["local"]:
        raise ValueError("data.local must be a path")
    resume = recipe.get("resume") or ""
    extend = recipe.get("extend", False)
    if type(extend) is not bool:
        raise ValueError("extend must be true or false")
    if extend and not resume:
        raise ValueError("extend requires resume")
    gpu = recipe.get("gpu", "L40S")
    if gpu not in GPUS:
        raise ValueError(f"gpu must be one of {GPUS}")
    if not isinstance(recipe["settings"], dict):
        raise ValueError("settings must be an object")
    settings = asdict(Settings(**recipe["settings"]))
    return {
        **{k: recipe[k] for k in ("name", "description", "provenance") if k in recipe},
        "run_id": run_id(recipe["run_id"]),
        "parent": checkpoint(recipe["parent"], "parent"),
        "data": dict(data),
        "resume": checkpoint(resume, "resume") if resume else "",
        "extend": extend,
        "gpu": gpu,
        "settings": settings,
    }


def load_recipe(path, run_id="", parent="", dataset="", data_id="", resume=""):
    """Load a recipe file and apply nonempty command-line overrides."""
    recipe = json.loads(Path(path).read_text())
    if dataset and data_id:
        raise ValueError("Pass either --dataset or --data-id, not both")
    overrides = {"run_id": run_id, "parent": parent, "resume": resume}
    recipe.update({k: v for k, v in overrides.items() if v})
    if dataset:
        recipe["data"] = {"local": dataset}
    if data_id:
        recipe["data"] = {"volume": data_id}
    return validate_recipe(recipe)


def parse_checkpoints(value):
    """`name=run/step-N.pt,...` into an ordered {name: checkpoint} mapping."""
    result = {}
    for item in value.split(","):
        name, sep, path = item.strip().partition("=")
        if not sep or name in result:
            raise ValueError("Expected unique name=run/step-NNNNNNNNN.pt entries")
        result[run_id(name, "checkpoint name")] = checkpoint(path)
    return result
