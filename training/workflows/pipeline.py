"""Pipelines: post-training stages run one after another, each starting from the previous
stage's final checkpoint. Nothing here imports Modal.

A pipeline is a workflow file with `"stage": "pipeline"`:

    {
      "stage": "pipeline",
      "name": "s0-then-sft",
      "from": "<run>/step-NNNNNNNNN.pt",
      "steps": [
        {"workflow": "sft/recipes/math-code-replay-20b.json"},
        {"branches": ["sft/recipes/a.json", "sft/recipes/b.json"]}
      ]
    }

`from` is the first stage's parent: a checkpoint of any stage (`trainer/checkpoints.py`),
so a pipeline can start after pretraining, mid-training, SFT or RL. Each step is one
stage workflow file (relative to `training/`), or `branches`: several run side by side
from the same parent, which only the last step may have. A stage's own `parent` and
`run_id` are replaced: the parent by the previous step's final checkpoint, the run ID by
`<pipeline run>-<step>-<file stem>`.

A launch that ends before the pipeline does (its 24-hour deadline, a failure) leaves
`/runs/<pipeline run>/pipeline.json` with every finished stage and its checkpoint;
launching the same pipeline file with the same run ID continues with the unfinished
stages. A stage that did not finish is rerun from its parent under `<run ID>-a<attempt>`.
"""

import hashlib
import json
from pathlib import Path

from .checks import checkpoint, run_id
from .stages import STAGES, validate

# Runs of one stage across relaunches: `<run ID>` and `<run ID>-a2` to `-a9`.
MAX_ATTEMPTS = 9
FIELDS = {"stage", "name", "description", "from", "steps"}


def stage_run(pipeline_run, index, path):
    return f"{pipeline_run}-{index + 1:02d}-{Path(path).stem}"


def attempt_run(name, attempt):
    return name if attempt == 1 else f"{name}-a{attempt}"


def load_pipeline(path, run, project_root=None):
    """The resolved plan of a pipeline file for pipeline run ID `run`: every stage's
    validated recipe, with the `from` checkpoint as the first step's parent."""
    project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    path = Path(path)
    raw = path.read_bytes()
    spec = json.loads(raw)
    if not isinstance(spec, dict) or set(spec) - FIELDS or spec.get("stage") != "pipeline":
        raise ValueError(f'A pipeline has "stage": "pipeline" and fields among {sorted(FIELDS)}')
    steps = spec.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("A pipeline needs steps")
    sources = {"pipeline": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}}
    plan = []
    for index, step in enumerate(steps):
        if (
            not isinstance(step, dict)
            or len(step) != 1
            or not set(step) <= {"workflow", "branches"}
        ):
            raise ValueError('A step is {"workflow": path} or {"branches": [paths]}')
        paths = [step["workflow"]] if "workflow" in step else step["branches"]
        if not isinstance(paths, list) or not paths or len(set(paths)) != len(paths):
            raise ValueError("Branches name distinct workflow files")
        if "branches" in step and index != len(steps) - 1:
            raise ValueError("Only the last step may branch")
        entries = []
        for relative in paths:
            file = (project_root / relative).resolve()
            if not file.is_relative_to(project_root):
                raise ValueError("A stage workflow escapes the project directory")
            stage = json.loads(file.read_text()).get("stage")
            if stage not in STAGES:
                raise ValueError(f"A pipeline step runs one of {sorted(STAGES)}")
            name = stage_run(run, index, relative)
            run_id(attempt_run(name, MAX_ATTEMPTS), "pipeline stage run ID")
            recipe = json.loads(file.read_text())
            recipe.update(run_id=name, parent=checkpoint(spec.get("from"), "from"))
            entries.append({"stage": stage, "recipe": validate(recipe)})
            sources[name] = {
                "path": relative,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            }
        plan.append(entries)
    return {
        **{k: spec[k] for k in ("name", "description") if k in spec},
        "stage": "pipeline",
        "run_id": run_id(run),
        "from": spec["from"],
        "steps": plan,
        "sources": sources,
    }
