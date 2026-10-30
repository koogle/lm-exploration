"""The data stage: prepare training data for a later pretraining or mid-training stage,
which names it in its `data` field.

- subword: `dataset_config` (relative to training/) is a `prepare_raw` dataset config,
  prepared into `<root>/<version>/snapshots/<stage run ID>` on the mixture data volume
  (`training_data.fast_raw_corpus`). Its consumer's settings (token target, context,
  preparation workers) size and check the preparation; with `overlap_preparation` the
  consumer starts training on the growing snapshot.
- hybrid: `recipe` (relative to training/, `hybrid/recipes/*.json`) is prepared into
  token shards under `<recipe name>/` on the hybrid data volume
  (`model_hybrid.modal_data`), with at most `parallel` shard workers.
"""

import json

from workflows.config import DatasetConfig, project_file

FIELDS = {"subword": {"dataset_config", "version"}, "hybrid": {"recipe", "parallel"}}


def resolve(entry, family, project_root):
    """The data stage's frozen inputs."""
    fields = set(entry) - {"stage", "name", "description"}
    if fields - FIELDS[family]:
        raise ValueError(f"A {family} data stage takes {sorted(FIELDS[family])}")
    if family == "subword":
        path = project_file(project_root, entry.get("dataset_config", ""))
        dataset = DatasetConfig(**json.loads(path.read_text()))
        if dataset.mode != "prepare_raw":
            raise ValueError("A data stage prepares a prepare_raw dataset config")
        version = entry.get("version")
        if not isinstance(version, str) or not version.replace("-", "").replace("_", "").isalnum():
            raise ValueError("A subword data stage names its version folder")
        return {"dataset_config": entry["dataset_config"], "version": version}
    path = project_file(project_root, entry.get("recipe", ""))
    recipe = json.loads(path.read_text())
    parallel = entry.get("parallel", 48)
    if type(parallel) is not int or parallel < 1:
        raise ValueError("parallel must be a positive integer")
    return {
        "recipe": entry["recipe"],
        # The path training containers read the recipe from.
        "container_recipe": "/project/model_hybrid/"
        + str(path.relative_to(project_root / "hybrid")),
        "recipe_name": recipe["name"],
        "parallel": parallel,
    }
