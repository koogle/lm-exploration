"""Export a trusted, self-describing checkpoint as an inference-only bundle."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_checkpoint(checkpoint, output, tokenizer, *, data_manifest=None):
    """Use checkpoint architecture metadata and current matching code, never old Git revisions."""
    checkpoint, output, tokenizer = Path(checkpoint), Path(output), Path(tokenizer)
    if output.exists():
        raise ValueError("Export output already exists")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "architecture" not in saved or "special_tokens" not in saved:
        raise ValueError(
            "Export requires a self-describing checkpoint with architecture and token IDs"
        )
    architecture = saved["architecture"]
    module = importlib.import_module(architecture["module"])
    definitions = {"model": module.__file__, **getattr(module, "MODEL_DEPENDENCIES", {})}
    for name, path in definitions.items():
        if digest(path) != saved["source_sha256"][name]:
            raise ValueError(f"Checkpoint model source differs from installed code: {name}")
    if digest(tokenizer) != saved["data_identity"]["tokenizer_sha256"]:
        raise ValueError("Checkpoint tokenizer checksum mismatch")
    if data_manifest and digest(data_manifest) != saved["data_identity"]["manifest_sha256"]:
        raise ValueError("Checkpoint data manifest checksum mismatch")

    model = module.GPT(module.ModelConfig(**architecture["config"])).cpu().eval()
    model.load_state_dict(saved["model"], strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != saved["parameters"]:
        raise ValueError("Checkpoint parameter count mismatch")
    if not all(torch.isfinite(t).all() for t in saved["model"].values()):
        raise ValueError("Checkpoint contains non-finite model tensors")
    ids = saved["special_tokens"]
    if any(
        type(ids[k]) is not int or not 0 <= ids[k] < model.config.vocab_size for k in ("bos", "eos")
    ):
        raise ValueError("Invalid checkpoint special-token IDs")
    if Tokenizer.from_file(str(tokenizer)).get_vocab_size() > model.config.vocab_size:
        raise ValueError("Tokenizer exceeds model vocabulary")

    # Include the selected architecture package and its declared local dependencies.
    dependencies = {Path(path).resolve() for path in definitions.values()}
    packages = set()
    for name, loaded in sys.modules.copy().items():
        path = getattr(loaded, "__file__", None)
        if isinstance(path, str) and Path(path).resolve() in dependencies:
            packages.add(name.split(".")[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        for name in sorted(packages):
            package = importlib.import_module(name)
            source = Path(next(iter(package.__path__)))
            for path in source.rglob("*.py"):
                target = bundle / name / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        weights = {k: v.contiguous().clone() for k, v in saved["model"].items()}
        save_file(weights, str(bundle / "model.safetensors"))
        restored = load_file(str(bundle / "model.safetensors"))
        if any(not torch.equal(weights[k], restored[k]) for k in weights):
            raise ValueError("Safetensors weight round-trip differs")
        sample = torch.tensor([[ids["bos"]]])
        with torch.inference_mode():
            before = model(sample)
            model.load_state_dict(restored, strict=True)
            after = model(sample)
        if not torch.isfinite(after).all() or not torch.equal(before, after):
            raise ValueError("Export logits are non-finite or differ after weight round-trip")
        (bundle / "config.json").write_text(
            json.dumps({"architecture": architecture, "special_tokens": ids}, indent=2) + "\n"
        )
        (bundle / "tokenizer").mkdir()
        shutil.copyfile(tokenizer, bundle / "tokenizer/tokenizer.json")
        if data_manifest:
            shutil.copyfile(data_manifest, bundle / "data-manifest.json")
        shutil.copyfile(Path(__file__).with_name("generate.py"), bundle / "generate.py")
        provenance = {
            "checkpoint_sha256": digest(checkpoint),
            "weights_sha256": digest(bundle / "model.safetensors"),
            "architecture": architecture,
            "parameters": parameters,
            "step": saved.get("step"),
            "tokens": saved.get("tokens"),
            "data_identity": saved["data_identity"],
            "model_source_sha256": {name: digest(path) for name, path in definitions.items()},
            "export_torch_version": str(torch.__version__),
        }
        (bundle / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (bundle / "requirements.txt").write_text(
            "".join(
                f"{name}=={importlib.metadata.version(name).split('+')[0]}\n"
                for name in ("torch", "safetensors", "tokenizers")
            )
        )
        (bundle / "README.md").write_text(
            "# Simple LLM inference bundle\n\n"
            "Custom PyTorch model; not a Transformers AutoModel package.\n"
            "This bundle excludes optimizer, RNG and data-cursor state and is not a resume checkpoint.\n\n"
            "```sh\npip install -r requirements.txt\npython generate.py --prompt Hello\n```\n"
        )
        bundle.rename(output)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted checkpoint only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON matching the checkpoint")
    parser.add_argument("--data-manifest", help="Optional matching provenance manifest")
    args = parser.parse_args()
    print(
        json.dumps(
            export_checkpoint(
                args.checkpoint, args.output, args.tokenizer, data_manifest=args.data_manifest
            )
        )
    )


if __name__ == "__main__":
    main()
