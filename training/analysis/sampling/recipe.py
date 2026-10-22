"""Sampling recipes: pinned models, prompts, decoding modes and limits (no torch)."""

import json
import re
from pathlib import Path

PRECISIONS = ("bf16-autocast", "fp32")
KINDS = ("checkpoint", "transformers")
MODE_DEFAULTS = {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}


def load_recipe(path):
    """Load and validate a recipe; resolve `prompt_file` into prompts.

    `prompt_file` names a task file (relative to the recipe) whose `tasks[].prompt`
    entries are appended to `prompts` under `prompt_group` (default "tasks").
    """
    path = Path(path)
    recipe = json.loads(path.read_text())
    if "prompt_file" in recipe:
        tasks = json.loads((path.parent / recipe["prompt_file"]).read_text())["tasks"]
        group = recipe.get("prompt_group", "tasks")
        recipe["prompts"] = [
            *recipe.get("prompts", []),
            *({"text": task["prompt"], "group": group} for task in tasks),
        ]
    validate(recipe)
    return recipe


def validate(recipe):
    """Reject unpinned models, unknown settings and unbounded generation."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", recipe["name"]):
        raise ValueError("recipe name must be a lowercase file stem")
    if recipe["precision"] not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if not 0 < recipe["max_new_tokens"] <= 1024:
        raise ValueError("max_new_tokens must be in 1..1024")
    if not recipe["prompts"] or not recipe["modes"] or not recipe["models"]:
        raise ValueError("prompts, modes and models must be non-empty")
    if len({model["kind"] for model in recipe["models"]}) != 1:
        raise ValueError("one recipe uses one model kind")
    for model in recipe["models"]:
        if model["kind"] not in KINDS:
            raise ValueError(f"model kind must be one of {KINDS}")
        revisions = [model["revision"]] if model["kind"] == "transformers" else []
        if model["kind"] == "checkpoint":
            revisions = [model["source_revision"], *(c["revision"] for c in model["checkpoints"])]
        if not all(re.fullmatch(r"[0-9a-f]{40}", r) for r in revisions):
            raise ValueError(f"{model['name']}: revisions must be full commit hashes")
    names = [mode["name"] for mode in recipe["modes"]]
    if len(set(names)) != len(names):
        raise ValueError("mode names must be unique")
    for mode in expand_modes(recipe):
        if (
            mode["temperature"] < 0
            or mode["top_k"] < 0
            or not 0 < mode["top_p"] <= 1
            or mode["repetition_penalty"] <= 0
        ):
            raise ValueError(f"{mode['name']}: invalid decoding settings")


def prompts(recipe):
    """(text, group) pairs; a plain string prompt has an empty group."""
    return [
        (p, "") if isinstance(p, str) else (p["text"], p.get("group", ""))
        for p in recipe["prompts"]
    ]


def expand_modes(recipe):
    """One decoding setting per seed offset; the sample seed is offset + prompt index."""
    for mode in recipe["modes"]:
        settings = {**MODE_DEFAULTS, **{k: v for k, v in mode.items() if k != "seed_offsets"}}
        for offset in mode.get("seed_offsets", [1000]):
            yield {**settings, "seed_offset": offset}
