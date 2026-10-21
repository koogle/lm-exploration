"""Export our trusted two-step DDP checkpoint as a standalone inference bundle."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from publication.artifacts import artifact_path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--artifacts", required=True, help="Verified dataset download directory")
    parser.add_argument("--training-report", required=True, help="Explicit smoke-run report path")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    # Only load the trusted checkpoint downloaded from our own Modal run volume.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert saved["format"] == "scholarly-ddp-v1"
    assert saved["step"] == 2 and saved["tokens"] == 121777
    revision = subprocess.check_output(["git", "rev-parse", "c69c38c"], cwd=root, text=True).strip()
    for package, folder in [("model_100m", "100m"), ("model_10m", "10m")]:
        (out / package).mkdir()
        for name in ["__init__.py", "model.py"]:
            contents = subprocess.check_output(
                ["git", "show", f"{revision}:training/{folder}/{name}"], cwd=root
            )
            (out / package / name).write_bytes(contents)
    assert digest(out / "model_100m/model.py") == saved["source_sha256"]["model"]
    sys.path.insert(0, str(out))
    from model_100m.model import GPT, ModelConfig

    cfg = {k: saved["config"][k] for k in ["n_layer", "n_head", "n_embd"]}
    cfg.update(vocab_size=32768, block_size=saved["config"]["context"], architecture_version=2)
    model = GPT(ModelConfig(**cfg))
    model.load_state_dict(saved["model"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == saved["parameters"]
    # Clone the tied output weight so safetensors contains every state_dict key.
    weights = {k: v.contiguous().clone() for k, v in saved["model"].items()}
    save_file(weights, str(out / "model.safetensors"))
    restored = load_file(str(out / "model.safetensors"))
    assert all(torch.equal(weights[k], restored[k]) for k in weights)
    model.load_state_dict(restored, strict=True)
    torch.set_num_threads(2)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([[1, 100, 200]]))
    assert torch.isfinite(logits).all()
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.copytree(root / "tokenizer/structured-code-v1", out / "tokenizer")
    assert digest(out / "tokenizer/tokenizer.json") == saved["data_identity"]["tokenizer_sha256"]
    report = Path(args.training_report)
    shutil.copyfile(report, out / "training-report.json")
    provenance = {k: v for k, v in saved.items() if k not in ["model", "optimizer", "rng", "data"]}
    provenance.update(
        source_commit=revision,
        checkpoint_sha256=digest(args.checkpoint),
        weights_sha256=digest(out / "model.safetensors"),
        strict_load_verified=True,
        weight_roundtrip_exact=True,
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "requirements.txt").write_text("torch==2.6.0\nsafetensors>=0.4\ntokenizers==0.22.2\n")
    shutil.copyfile(
        artifact_path(args.artifacts, "publication/scholarly-124m-step2-card.md"), out / "README.md"
    )
    shutil.copyfile(root / "publication/generate.py", out / "generate.py")
    shutil.copyfile(args.data_manifest, out / "data-manifest.json")
    assert digest(out / "data-manifest.json") == saved["data_identity"]["manifest_sha256"]
    print(
        json.dumps(
            {
                "parameters": saved["parameters"],
                "step": saved["step"],
                "tokens": saved["tokens"],
                "output": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
