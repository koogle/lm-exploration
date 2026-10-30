"""Validate RL workflow files (`"stage": "rl"`) without Modal.

A recipe names one run: `run_id`, `parent` (an SFT checkpoint, `run/step-N.pt` on the run
volume), `pool` (an `rl_pool` data ID) and `settings` (`trainer/rl/settings.py`), plus an
optional `name` and `description`.
"""

import json
from dataclasses import asdict
from pathlib import Path

from trainer.rl.settings import Settings
from trainer.sft_recipe import checkpoint, run_id

FIELDS = {"stage", "name", "description", "run_id", "parent", "pool", "settings"}


def validate_recipe(recipe):
    """A normalized copy with every setting filled in, or ValueError."""
    if not isinstance(recipe, dict) or set(recipe) - FIELDS:
        raise ValueError(f"Recipe fields must be among {sorted(FIELDS)}")
    if recipe.get("stage") != "rl":
        raise ValueError('An RL recipe has "stage": "rl"')
    settings = Settings.from_dict(recipe["settings"])
    if settings.trajectories.source == "file":
        run_id(settings.trajectories.data, "trajectories.data")
    return {
        **{k: recipe[k] for k in ("name", "description") if k in recipe},
        "stage": "rl",
        "run_id": run_id(recipe["run_id"]),
        "parent": checkpoint(recipe["parent"], "parent"),
        "pool": run_id(recipe["pool"], "pool"),
        "settings": asdict(settings),
    }


def load_recipe(path, run=""):
    """Load a recipe file; a nonempty `run` replaces its run ID."""
    recipe = json.loads(Path(path).read_text())
    if run:
        recipe["run_id"] = run
    return validate_recipe(recipe)
