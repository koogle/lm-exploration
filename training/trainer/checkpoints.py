"""Checkpoints of every training stage, read one way. Only read our own checkpoints: they
are pickles.

Every checkpoint records `architecture`: the model module (which names the family,
`trainer/families.py`) and its config. Two layouts:

- Pretraining (`trainer/loop.py`): `architecture`, `source_sha256` and `data_identity`;
  the subword family copies its model packages to `RUN-sNN/source`.
- Post-training (SFT, RL and any later stage): `stage`, `architecture` and `identity`
  (with `source_sha256`, `tokenizer_sha256` and the parent's `parent_sha256`); the
  family's source files are copied beside the checkpoint.

`describe` gives what a stage needs to start from any checkpoint, whatever stage and
family wrote it, and `parent` the family's adapter with it. `load` builds a subword
model with the checkpoint's own recorded code, checked against its source hashes, so
any checkout can load any subword checkpoint.
"""

import importlib
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from training_data.subword import digest


@dataclass(frozen=True)
class Description:
    """The stage that wrote a checkpoint, its model module and config, the source hashes
    of its model code, its tokenizer's hash, and whether it was trained to answer
    instructions: any post-training stage, or the pretraining loop on a chat-format recipe
    (hybrid chat SFT is a `midtrain` stage of that loop)."""

    stage: str
    module: str
    model_config: dict
    source_sha256: dict
    tokenizer_sha256: str
    instructed: bool


def describe(saved):
    """The `Description` of a loaded checkpoint dictionary of either layout."""
    if "architecture" not in saved:
        raise ValueError("Not a checkpoint of our trainers")
    module, config = saved["architecture"]["module"], saved["architecture"]["config"]
    if "stage" in saved:
        identity = saved["identity"]
        stage, sources, tokenizer = saved["stage"], identity["source_sha256"], identity
        instructed = True
    else:
        stage, sources, tokenizer = "pretrain", saved["source_sha256"], saved["data_identity"]
        recipe = tokenizer.get("recipe")
        instructed = isinstance(recipe, dict) and recipe.get("format") == "chat-sft"
    return Description(stage, module, config, sources, tokenizer["tokenizer_sha256"], instructed)


def parent(path):
    """(checkpoint dictionary, `Description`, family adapter) of a parent to continue
    training, from any stage of any family.

    The parent's model code may be an earlier version of its family's: the trainer
    loads its weights strictly into the current model, so they must fit, and records
    the parent's source hashes (`Description.source_sha256`) beside its own."""
    import torch

    from trainer import families

    # Only load our trusted private run-volume checkpoints, never dataset-supplied files.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    description = describe(saved)
    family = families.adapter(description.module)
    if description.tokenizer_sha256 != family.tokenizer_sha256:
        raise ValueError("Parent uses another tokenizer than its family")
    return saved, description, family


def import_model(source, name):
    """Import `name` from `source` only, dropping any previously loaded run's model code."""
    for loaded in [n for n in sys.modules if n.split(".")[0] in ("model_100m", "model_10m")]:
        del sys.modules[loaded]
    sys.path[:] = [p for p in sys.path if not p.endswith("/source")]
    sys.path.insert(0, str(source))
    module = importlib.import_module(name)
    if not Path(module.__file__).is_relative_to(source):
        raise ValueError(f"Model code must come from the checkpoint's own run: {module.__file__}")
    return module


def recorded_code(path, saved, description):
    """The model module a checkpoint's run recorded, checked against its source hashes."""
    if description.stage == "pretrain":
        module = import_model(path.parent / "source", description.module)
        definitions = {"model": module.__file__, **getattr(module, "MODEL_DEPENDENCIES", {})}
        for key, file in definitions.items():
            if digest(file) != description.source_sha256[key]:
                raise ValueError(f"Model source differs from the checkpoint: {key}")
        return module
    run = path.parent
    source = Path(tempfile.mkdtemp(prefix="checkpoint-")) / "source"
    for package, file, key in (
        ("model_100m", "model.py", "model"),
        ("model_10m", "shared_model.py", "shared_model"),
    ):
        if digest(run / file) != description.source_sha256[key]:
            raise ValueError(f"Model source differs from the checkpoint: {key}")
        (source / package).mkdir(parents=True)
        (source / package / "__init__.py").write_text("")
        (source / package / "model.py").write_bytes((run / file).read_bytes())
    return import_model(source, description.module)


def load(path, device="cpu"):
    """A verified checkpoint: SimpleNamespace(model, module, config, tokenizer, special, info)."""
    import torch
    from tokenizers import Tokenizer

    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    description = describe(saved)
    if description.module != "model_100m.model":
        raise ValueError("Recorded-code loading is for subword checkpoints")
    module = recorded_code(path, saved, description)
    tokenizer_path = path.parent / "tokenizer" / "tokenizer.json"
    if digest(tokenizer_path) != description.tokenizer_sha256:
        raise ValueError("Tokenizer differs from the checkpoint")
    model = module.GPT(module.ModelConfig(**description.model_config))
    model.load_state_dict(saved["model"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Non-finite parameters")
    info = {
        "checkpoint_sha256": digest(path),
        "stage": description.stage,
        "format": saved.get("format"),
        "step": saved.get("step"),
        "trained_tokens": saved.get("tokens"),
        "sft_answer_tokens": saved.get("answer_tokens"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_source_sha256": description.source_sha256,
        "tokenizer_sha256": description.tokenizer_sha256,
    }
    return SimpleNamespace(
        model=model.to(device).eval(),
        module=module,
        config=description.model_config,
        tokenizer=Tokenizer.from_file(str(tokenizer_path)),
        tokenizer_path=tokenizer_path,
        special=saved["special_tokens"],
        state=saved["model"],
        info=info,
    )
