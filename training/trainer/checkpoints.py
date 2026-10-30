"""Checkpoints of every training stage, read one way. Only read our own checkpoints: they
are pickles.

Two layouts:

- Pretraining (`trainer/loop.py`): `architecture` (model module and config),
  `source_sha256` and `data_identity`; the model's packages are copied to
  `RUN-sNN/source`.
- Post-training (SFT, RL and any later stage): `stage`, `model_config` and `identity`
  (with `source_sha256`, `tokenizer_sha256` and the parent's `parent_sha256`);
  `model.py` and `shared_model.py` are copied beside the checkpoint.

Both keep the tokenizer in the run's `tokenizer/` folder. `describe` gives what a stage
needs to start from any checkpoint, whatever stage wrote it; `load` builds the model with
the checkpoint's own recorded code, checked against its source hashes, so any checkout
can load any checkpoint.
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
    of its model code and its tokenizer's hash."""

    stage: str
    module: str
    model_config: dict
    source_sha256: dict
    tokenizer_sha256: str


def describe(saved):
    """The `Description` of a loaded checkpoint dictionary of either layout."""
    if "architecture" in saved:
        return Description(
            "pretrain",
            saved["architecture"]["module"],
            saved["architecture"]["config"],
            saved["source_sha256"],
            saved["data_identity"]["tokenizer_sha256"],
        )
    if "identity" in saved and "stage" in saved:
        identity = saved["identity"]
        return Description(
            saved["stage"],
            "model_100m.model",
            saved["model_config"],
            identity["source_sha256"],
            identity["tokenizer_sha256"],
        )
    raise ValueError("Not a checkpoint of our trainers")


def parent(path, model_module, tokenizer_sha256):
    """(checkpoint dictionary, `Description`) of a parent to continue training with
    `model_module`'s code and the tokenizer of `tokenizer_sha256`, from any stage."""
    import torch

    # Only load our trusted private run-volume checkpoints, never dataset-supplied files.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    description = describe(saved)
    if description.module != model_module.__name__ or description.source_sha256["model"] != digest(
        model_module.__file__
    ):
        raise ValueError("Model implementation differs from the parent checkpoint")
    if description.tokenizer_sha256 != tokenizer_sha256:
        raise ValueError("Parent uses another tokenizer")
    return saved, description


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
