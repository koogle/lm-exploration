"""Load our pretraining and SFT checkpoints with the model code each run recorded.

Pretraining runs copy their packages to `RUN-sNN/source`; SFT runs
(`trainer/sft.py`) copy `model.py` and `shared_model.py` beside their checkpoints.
Each file is checked against the source hash the checkpoint stored, so any checkout can
load any checkpoint. Only load our own checkpoints: they are pickles.
"""

import importlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from training_data.subword import digest

SFT_FORMATS = ("math-sft-v1", "repair-sft-v2", "code-sft-v1", "mixture-sft-v1")
# Padding sits after every scored position; PAD is ID 0 in our tokenizers.
PAD_TOKEN_ID = 0


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


def pretraining_code(path, saved):
    """(model module, config, tokenizer SHA256, source hashes) of a pretraining checkpoint."""
    architecture = saved["architecture"]
    module = import_model(path.parent / "source", architecture["module"])
    definitions = {"model": module.__file__, **getattr(module, "MODEL_DEPENDENCIES", {})}
    for key, file in definitions.items():
        if digest(file) != saved["source_sha256"][key]:
            raise ValueError(f"Model source differs from the checkpoint: {key}")
    tokenizer = saved["data_identity"]["tokenizer_sha256"]
    return module, architecture["config"], tokenizer, saved["source_sha256"]


def sft_code(path, saved):
    """The same for an SFT checkpoint, rebuilt as packages in a temporary directory."""
    run, identity = path.parent, saved["identity"]
    source = Path(tempfile.mkdtemp(prefix="sft-")) / "source"
    for package, file, key in (
        ("model_100m", "model.py", "model"),
        ("model_10m", "shared_model.py", "shared_model"),
    ):
        if digest(run / file) != identity["source_sha256"][key]:
            raise ValueError(f"Model source differs from the checkpoint: {key}")
        (source / package).mkdir(parents=True)
        (source / package / "__init__.py").write_text("")
        (source / package / "model.py").write_bytes((run / file).read_bytes())
    module = import_model(source, "model_100m.model")
    return module, saved["model_config"], identity["tokenizer_sha256"], identity["source_sha256"]


def load(path, device="cpu"):
    """A verified checkpoint: SimpleNamespace(model, module, config, tokenizer, special, info)."""
    import torch
    from tokenizers import Tokenizer

    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    code = sft_code if saved.get("format") in SFT_FORMATS else pretraining_code
    module, config, tokenizer_sha256, source_sha256 = code(path, saved)
    tokenizer_path = path.parent / "tokenizer" / "tokenizer.json"
    if digest(tokenizer_path) != tokenizer_sha256:
        raise ValueError("Tokenizer differs from the checkpoint")
    model = module.GPT(module.ModelConfig(**config))
    model.load_state_dict(saved["model"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Non-finite parameters")
    info = {
        "checkpoint_sha256": digest(path),
        "format": saved.get("format"),
        "step": saved.get("step"),
        "trained_tokens": saved.get("tokens"),
        "sft_answer_tokens": saved.get("answer_tokens"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_source_sha256": source_sha256,
        "tokenizer_sha256": tokenizer_sha256,
    }
    return SimpleNamespace(
        model=model.to(device).eval(),
        module=module,
        config=config,
        tokenizer=Tokenizer.from_file(str(tokenizer_path)),
        tokenizer_path=tokenizer_path,
        special=saved["special_tokens"],
        state=saved["model"],
        info=info,
    )


def scorer(checkpoint):
    """The model interface `tasks.py` expects, running in BF16 autocast on CUDA."""
    import torch

    def forward(tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return checkpoint.model(tokens)

    tokenizer = checkpoint.tokenizer
    return SimpleNamespace(
        forward=forward,
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        bos_token_id=checkpoint.special["bos"],
        eos_token_id=checkpoint.special["eos"],
        pad_token_id=PAD_TOKEN_ID,
        block_size=checkpoint.config["block_size"],
        device="cuda",
    )
