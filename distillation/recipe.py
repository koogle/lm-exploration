"""Validate distillation recipes for `modal_app.py` and `modal_vllm.py` (no Modal, no torch).

A recipe is a JSON object describing one job. `kind` selects the job; the other fields name
every input explicitly, so launchers hold no run IDs or paths:

- `byte`: online first-byte-marginal KL from a Hugging Face teacher (`train.py`), one L40S.
- `shared`: shared-tokenizer full-vocabulary KL (`shared_train.py`), one H100.
- `students`: fresh students on a completed curriculum corpus (`curriculum_train.py`), one H100.
- `curriculum`: teacher generation of verified curriculum examples (`modal_vllm.py`), one H100.
- `evaluate`: CE of several checkpoints on one fixed code holdout (`evaluate.py`), one L4.
- `generations`: fixed-prompt continuations from distillation checkpoints, one L4, decoded
  by the sampling toolkit (`training/analysis/sampling/sampler.py`).

Paths are relative to a Modal volume: `parent` to `lm-exploration-training`, `data` to
`lm-exploration-mixture-data`, and run IDs, `resume`, `corpus` and generation checkpoints to
`lm-exploration-distillation`. Evaluation checkpoints carry a `training:` or `distillation:`
volume prefix. Command-line overrides replace recipe values.
"""

import json
import re
from pathlib import Path

RUN_ID = re.compile(r"[a-zA-Z0-9_-]{1,80}")
CHECKPOINT = re.compile(r"[a-zA-Z0-9_-]{1,80}(/[a-zA-Z0-9_-]{1,80})?/step-[0-9]{9}\.pt")
DATA_PATH = re.compile(r"[a-zA-Z0-9_.-]+(/[a-zA-Z0-9_.-]+)*")
REVISION = re.compile(r"[0-9a-f]{40}")
REPO = re.compile(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+")
VOLUMES = ("training", "distillation")
PRECISIONS = ("bf16-autocast", "fp32")
COMMON = {"name", "description", "provenance", "kind", "run_id"}
FIELDS = {
    "byte": {"teacher", "parent", "data", "resume", "target_tokens", "max_seconds"},
    "shared": {"teacher", "parent", "data", "resume", "target_tokens", "max_seconds"},
    "students": {"corpus"},
    "curriculum": {"teacher", "count", "sampling"},
    "evaluate": {
        "checkpoints",
        "data",
        "context",
        "max_documents",
        "windows_per_document",
    },
    "generations": {"checkpoints", "precision", "max_new_tokens", "prompts", "modes"},
}
REQUIRED = {
    "byte": {"teacher", "parent", "data", "target_tokens", "max_seconds"},
    "shared": {"teacher", "parent", "data", "target_tokens", "max_seconds"},
    "students": {"corpus"},
    "curriculum": {"teacher", "count"},
    "evaluate": {"checkpoints", "data"},
    "generations": {"checkpoints", "precision", "max_new_tokens", "prompts", "modes"},
}
# Cooperative training-time caps; each stays below its launcher's function timeout.
MAX_SECONDS = {"byte": 1200, "shared": 82000}
CURRICULUM_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 1024,
    "seed": 42,
}


def run_id(value, field="run_id"):
    if not isinstance(value, str) or not RUN_ID.fullmatch(value):
        raise ValueError(f"{field} must match {RUN_ID.pattern}")
    return value


def checkpoint(value, field="checkpoint"):
    """Volume-relative `run/step-N.pt` or `run/sub/step-N.pt`; no absolute or parent paths."""
    if not isinstance(value, str) or not CHECKPOINT.fullmatch(value):
        raise ValueError(f"{field} must be run-id[/folder]/step-NNNNNNNNN.pt")
    return value


def volume_checkpoint(value):
    """`training:<checkpoint>` or `distillation:<checkpoint>`."""
    volume, sep, path = str(value).partition(":")
    if not sep or volume not in VOLUMES:
        raise ValueError(f"checkpoint {value!r} needs a volume prefix from {VOLUMES}")
    return f"{volume}:{checkpoint(path)}"


def data_path(value, field):
    if not isinstance(value, str) or not DATA_PATH.fullmatch(value) or ".." in value:
        raise ValueError(f"{field} must be a relative path without '..'")
    return value


def teacher(value):
    if not isinstance(value, dict) or set(value) != {"repo", "revision"}:
        raise ValueError('teacher must be {"repo": ..., "revision": <full commit>}')
    if not REPO.fullmatch(str(value["repo"])):
        raise ValueError("teacher.repo must be owner/name")
    if not REVISION.fullmatch(str(value["revision"])):
        raise ValueError("teacher.revision must be a full commit hash")
    return dict(value)


def positive(value, field, maximum=None):
    if type(value) is not int or value < 1 or (maximum and value > maximum):
        bound = f" and at most {maximum}" if maximum else ""
        raise ValueError(f"{field} must be a positive integer{bound}")
    return value


def generation_settings(recipe):
    """Prompt/mode checks matching the sampling toolkit's recipe format."""
    if recipe["precision"] not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    positive(recipe["max_new_tokens"], "max_new_tokens", 1024)
    prompts = recipe["prompts"]
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a nonempty list")
    for prompt in prompts:
        text = prompt if isinstance(prompt, str) else prompt.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("each prompt is a string or {text, group}")
    modes = recipe["modes"]
    if not isinstance(modes, list) or not modes:
        raise ValueError("modes must be a nonempty list")
    names = [mode.get("name") for mode in modes]
    if len(set(names)) != len(names) or not all(names):
        raise ValueError("modes need unique names")
    for mode in modes:
        if mode.get("temperature", 0) < 0 or not 0 < mode.get("top_p", 1) <= 1:
            raise ValueError(f"{mode['name']}: invalid decoding settings")


def validate_recipe(recipe):
    """Return a normalized copy, or raise. Safe to call locally and remotely."""
    if not isinstance(recipe, dict):
        raise TypeError("Recipe must be a JSON object")
    kind = recipe.get("kind")
    if kind not in FIELDS:
        raise ValueError(f"kind must be one of {sorted(FIELDS)}")
    unknown = set(recipe) - COMMON - FIELDS[kind]
    if unknown:
        raise ValueError(f"Unknown {kind} recipe fields: {sorted(unknown)}")
    missing = {"run_id"} | REQUIRED[kind]
    missing = sorted(field for field in missing if recipe.get(field) in (None, "", []))
    if missing:
        raise ValueError(f"{kind} recipe needs explicit {missing}")
    spec = dict(recipe)
    run_id(spec["run_id"])
    if "teacher" in spec:
        spec["teacher"] = teacher(spec["teacher"])
    if kind in ("byte", "shared", "evaluate"):
        data = spec["data"]
        if not isinstance(data, dict) or set(data) != {"manifest", "database"}:
            raise ValueError('data must be {"manifest": path, "database": path}')
        spec["data"] = {k: data_path(v, f"data.{k}") for k, v in data.items()}
    if kind in ("byte", "shared"):
        spec["parent"] = checkpoint(spec["parent"], "parent")
        spec["resume"] = (
            checkpoint(spec["resume"], "resume") if spec.get("resume") else ""
        )
        positive(spec["target_tokens"], "target_tokens")
        positive(spec["max_seconds"], "max_seconds", MAX_SECONDS[kind])
    if kind == "students":
        run_id(spec["corpus"], "corpus")
    if kind == "curriculum":
        positive(spec["count"], "count", 4096)
        if spec["count"] < 32:
            raise ValueError("count must be at least 32")
        sampling = spec.get("sampling", {})
        if set(sampling) - set(CURRICULUM_SAMPLING):
            raise ValueError(
                f"sampling keys must be among {sorted(CURRICULUM_SAMPLING)}"
            )
        spec["sampling"] = {**CURRICULUM_SAMPLING, **sampling}
    if kind == "evaluate":
        spec["checkpoints"] = [volume_checkpoint(c) for c in spec["checkpoints"]]
        spec["context"] = positive(spec.get("context", 256), "context", 8192)
        spec["max_documents"] = positive(
            spec.get("max_documents", 512), "max_documents"
        )
        spec["windows_per_document"] = positive(
            spec.get("windows_per_document", 8), "windows_per_document"
        )
    if kind == "generations":
        spec["checkpoints"] = [checkpoint(c) for c in spec["checkpoints"]]
        generation_settings(spec)
    return spec


def load_recipe(path, run_id="", parent="", resume=""):
    """Load a recipe file and apply nonempty command-line overrides."""
    recipe = json.loads(Path(path).read_text())
    overrides = {"run_id": run_id, "parent": parent, "resume": resume}
    for key, value in overrides.items():
        if value:
            if key != "run_id" and recipe.get("kind") not in ("byte", "shared"):
                raise ValueError(f"--{key} applies only to byte and shared recipes")
            recipe[key] = value
    return validate_recipe(recipe)
