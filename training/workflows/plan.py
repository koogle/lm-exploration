"""A workflow: the stages that make a model, from data preparation to RL, in one file.

    {
      "name": "hybrid-720m-v2",
      "family": "hybrid",
      "from": "<run>/step-NNNNNNNNN.pt",
      "stages": [
        {"stage": "data", "name": "corpus", ...},
        {"stage": "pretrain", "name": "base", "data": "corpus", ...},
        {"stage": "midtrain", "name": "long", "data": "...", ...},
        {"stage": "sft", "name": "chat", ...},
        {"stage": "rl", "branches": [{"name": "grpo", ...}, {"name": "awr", ...}]}
      ]
    }

Every stage of a `family` (`subword` or `hybrid`) has a unique `name`
(`workflows/stages/` describes each kind). Training stages run in order, each starting
from the previous one's final checkpoint; the first training stage after (or instead of)
pretraining starts from `from`, a checkpoint of any stage. A `data` stage feeds the
pretraining or mid-training stage that names it in `data`. The last stage may list
`branches`: stages of one kind that run side by side from the same parent.

A workflow may hold any part of this: only data and pretraining, or only SFT and RL from
a pretrained checkpoint. A launch may also run part of a workflow: `--start` and `--end`
select stages by name, and `--checkpoint` gives the parent of the first selected stage
when the stage before it has not finished in this workflow run.

Each stage runs as `<workflow run>-<stage name>`. Every launch records the stages that
finished, with their outputs, in `/runs/<workflow run>/stages.json`; launching the same
workflow and run ID again continues with the unfinished stages, so a long pretraining
stage continues across 24-hour launches.
"""

import hashlib
import json
from pathlib import Path

from workflows.checks import checkpoint, parent_path, run_id
from workflows.config import FAMILIES
from workflows.stages import (
    FAMILY_POST_TRAINING,
    KINDS,
    POST_TRAINING,
    TRAINING,
    data,
    pretrain,
    validate,
)

FIELDS = {"name", "description", "family", "from", "stages"}
# Suffixes a stage's run ID may gain: relaunch attempts and pretraining sessions.
LONGEST_SUFFIX = "-a9-s99"


def stage_run(workflow_run, name):
    return f"{workflow_run}-{name}"


def load_workflow(path, run, project_root=None):
    """The resolved plan of the workflow file at `path` for workflow run ID `run`."""
    project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    raw = Path(path).read_bytes()
    spec = json.loads(raw)
    if not isinstance(spec, dict) or set(spec) - FIELDS:
        raise ValueError(f"A workflow has fields among {sorted(FIELDS)}")
    family = spec.get("family")
    if family not in FAMILIES:
        raise ValueError(f"A workflow names its family, one of {FAMILIES}")
    entries = spec.get("stages")
    if not isinstance(entries, list) or not entries:
        raise ValueError("A workflow lists its stages")
    start = spec.get("from")
    if start is not None:
        parent_path(start, "from")
    run_id(run)
    stages, names = [], set()
    data_stages = {}
    previous = None  # the training stage the next one continues
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("stage") not in KINDS:
            raise ValueError(f"Every stage names its kind, one of {KINDS}")
        group = entry.get("branches")
        if group is not None:
            if index != len(entries) - 1 or entry["stage"] not in TRAINING:
                raise ValueError("Only the last stage, a training stage, may branch")
            if set(entry) != {"stage", "branches"} or not isinstance(group, list) or not group:
                raise ValueError('A branching stage is {"stage": kind, "branches": [stages]}')
            members = [{"stage": entry["stage"], **member} for member in group]
        else:
            members = [entry]
        for member in members:
            name = member.get("name")
            run_id(name, "stage name")
            if name in names:
                raise ValueError(f"Stage names must be unique: {name}")
            names.add(name)
            run_id(stage_run(run, name) + LONGEST_SUFFIX, f"run ID of stage {name}")
            stages.append(
                resolve_stage(member, family, previous, data_stages, start, run, project_root)
            )
        if entry["stage"] == "data":
            data_stages[stages[-1]["name"]] = stages[-1]
        elif entry["stage"] in TRAINING and group is None:
            previous = stages[-1]["name"]
    plan = {
        **{k: spec[k] for k in ("name", "description") if k in spec},
        "family": family,
        "run_id": run,
        "from": start,
        "stages": stages,
    }
    plan["sha256"] = hashlib.sha256(raw + json.dumps(stages, sort_keys=True).encode()).hexdigest()
    return plan


def resolve_stage(entry, family, previous, data_stages, start, run, project_root):
    kind, name = entry["stage"], entry["name"]
    stage = {"stage": kind, "name": name, "run_id": stage_run(run, name)}
    if kind == "data":
        return {**stage, "spec": data.resolve(entry, family, project_root)}
    # The parent: the previous training stage of the workflow, else `from`.
    if kind == "pretrain":
        if previous is not None:
            raise ValueError("Pretraining starts a model; continue one with midtrain")
        stage["parent"] = None
    else:
        stage["parent"] = previous or "from"
        if previous is None and start is None:
            raise ValueError(f"Stage {name} continues a checkpoint: give the workflow `from`")
    if kind in ("pretrain", "midtrain"):
        source = entry.get("data")
        if source is not None and source not in data_stages:
            raise ValueError(f"Stage {name} names an unknown data stage {source}")
        stage["data"] = source
        resolved = data_stages[source]["spec"] if source else None
        fields = {k: v for k, v in entry.items() if k not in ("stage", "name", "data", "resume")}
        # An exact resume of a pretraining checkpoint, such as a WSD decay branch.
        stage["resume"] = entry.get("resume", "")
        if stage["resume"]:
            if kind != "pretrain":
                raise ValueError("Only pretraining resumes a checkpoint exactly")
            checkpoint(stage["resume"], "resume")
        stage["spec"] = pretrain.resolve(fields, family, resolved, project_root)
        return stage
    if kind not in POST_TRAINING or kind not in FAMILY_POST_TRAINING[family]:
        raise ValueError(f"{family} has no {kind} stage yet")
    if {"run_id", "parent"} & set(entry):
        raise ValueError(f"Stage {name}: the workflow sets run_id and parent")
    fields = {k: v for k, v in entry.items() if k != "name"}
    # The parent and run ID come from the workflow; validate with stand-ins.
    spec = validate({**fields, "run_id": stage["run_id"], "parent": "parent/step-000000000.pt"})
    stage["spec"] = {k: v for k, v in spec.items() if k not in ("run_id", "parent")}
    return stage


def select(plan, first="", last=""):
    """The names of the stages `--start first --end last` runs, in order."""
    names = [stage["name"] for stage in plan["stages"]]
    for value in (first, last):
        if value and value not in names:
            raise ValueError(f"No stage named {value}")
    begin = names.index(first) if first else 0
    end = names.index(last) + 1 if last else len(names)
    if begin >= end:
        raise ValueError("--start names a stage after --end")
    return names[begin:end]
