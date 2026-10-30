"""The stages a workflow lists, by kind (`workflows/plan.py` resolves whole workflows):

- `data`: prepare training data (`data.py`).
- `pretrain`: train a model from scratch; `midtrain`: continue training a parent
  checkpoint on new data, context or schedule (`pretrain.py`).
- `sft`, `rl`: post-train a parent checkpoint (`sft.py`, `rl.py`).

A post-training stage entry holds the stage's own inputs and its trainer's `settings`;
the workflow supplies its `run_id` and its `parent`, a checkpoint of any stage
(`trainer/checkpoints.py`). `POST_TRAINING` maps each post-training kind to its module,
which declares `FIELDS` (its own inputs, such as SFT's `data` or RL's `pool`), `inputs(spec)` (those
inputs, checked and with defaults) and `settings(values)` (its trainer's settings with
every default filled in). Trainers know nothing about workflow files.
"""

from workflows.checks import checkpoint, run_id
from workflows.stages import rl, sft

KINDS = ("data", "pretrain", "midtrain", "sft", "rl")
# Stages that train a model and write checkpoints.
TRAINING = ("pretrain", "midtrain", "sft", "rl")
POST_TRAINING = {"sft": sft, "rl": rl}
COMMON = {"stage", "name", "description", "provenance", "run_id", "parent", "settings"}


def validate(spec):
    """A post-training stage's normalized inputs with its `run_id` and `parent`, or
    ValueError. Safe to call locally and remotely."""
    if not isinstance(spec, dict) or spec.get("stage") not in POST_TRAINING:
        raise ValueError(f"A post-training stage is one of {sorted(POST_TRAINING)}")
    stage = POST_TRAINING[spec["stage"]]
    unknown = set(spec) - COMMON - stage.FIELDS
    if unknown:
        raise ValueError(f"Unknown {spec['stage']} fields: {sorted(unknown)}")
    for field in ("run_id", "parent", "settings"):
        if not spec.get(field):
            raise ValueError(f"A {spec['stage']} stage needs {field}")
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
