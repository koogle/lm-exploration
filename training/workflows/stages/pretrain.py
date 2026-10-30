"""Pretraining and mid-training stages: one trainer (`trainer/loop.py` with the
workflow family's adapter), sessions resumed across 24-hour launches.

An entry holds the execution settings of `workflows.config.PretrainConfig` (GPUs,
sessions, qualification, and for subword the token target and data review) and
`model_config`, plus its data:

- `data`: the name of an earlier data stage of the workflow, or
- subword `dataset_config`: a prepared `snapshot` dataset config (relative to training/);
  a hybrid run config otherwise names its own prepared recipe.

`midtrain` continues a parent checkpoint (the previous training stage's, or the
workflow's `from`): subword seeds a new phase with the parent's weights and optimizer
state (`trainer/continued_pretraining.py`, same architecture and GPU count); hybrid starts
from the parent's exported weights (`init_from`) with a fresh optimizer, so its run
config may change the context length or drop MTP heads.
"""

from workflows.config import pretrain_bundle


def resolve(entry, family, data, project_root):
    """The stage's frozen bundle; `data` is the resolved data stage it names, if any."""
    spec = dict(entry)
    if data is not None and family == "subword":
        if "dataset_config" in spec:
            raise ValueError("A stage takes either data or dataset_config")
        spec["dataset_config"] = data["dataset_config"]
    dataset_config = spec.pop("dataset_config", "")
    bundle = pretrain_bundle(spec, family, dataset_config, project_root)
    if family == "hybrid" and data is not None:
        bundle["model"]["recipe"] = data["container_recipe"]
    if family == "subword" and data is None and bundle["dataset"]["mode"] != "snapshot":
        raise ValueError("A stage without a data stage trains on a prepared snapshot")
    return bundle
