"""Export the trusted completed 9.85M model; never publish optimizer/RNG state."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from training_data.subword import digest

from publication.artifacts import artifact_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data-manifest", required=True)
    p.add_argument("--artifacts", required=True, help="Verified dataset download directory")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert saved["tokens"] == 4999934057 and saved["step"] == 72805
    assert sum(saved["data"]["loss_token_exposures"].values()) == saved["tokens"]
    revision = subprocess.check_output(["git", "rev-parse", "4d24240"], cwd=root, text=True).strip()
    for package, folder in [("model_100m", "100m"), ("model_10m", "10m")]:
        (out / package).mkdir()
        for name in ("__init__.py", "model.py"):
            (out / package / name).write_bytes(
                subprocess.check_output(
                    ["git", "show", f"{revision}:training/{folder}/{name}"], cwd=root
                )
            )
    assert digest(out / "model_100m/model.py") == saved["source_sha256"]["model"]
    assert digest(args.data_manifest) == saved["data_identity"]["manifest_sha256"]
    sys.path.insert(0, str(out))
    from model_100m.model import GPT, ModelConfig

    cfg = {k: saved["config"][k] for k in ("n_layer", "n_head", "n_embd")}
    cfg.update(vocab_size=32768, block_size=2048, architecture_version=2)
    torch.set_num_threads(2)
    model = GPT(ModelConfig(**cfg)).eval()
    model.load_state_dict(saved["model"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 9849536
    assert all(torch.isfinite(v).all() for v in saved["model"].values())
    weights = {k: v.contiguous().clone() for k, v in saved["model"].items()}
    save_file(weights, str(out / "model.safetensors"))
    restored = load_file(str(out / "model.safetensors"))
    assert all(torch.equal(weights[k], restored[k]) for k in weights)
    x = torch.tensor([[1, 100, 200, 300]])
    with torch.no_grad():
        before = model(x)
    model.load_state_dict(restored, strict=True)
    with torch.no_grad():
        after = model(x)
    assert torch.equal(before, after) and torch.isfinite(after).all()
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.copytree(root / "tokenizer/structured-code-v1", out / "tokenizer")
    shutil.copyfile(args.data_manifest, out / "data-manifest.json")
    provenance = {k: v for k, v in saved.items() if k not in ("model", "optimizer", "rng", "data")}
    provenance.update(
        source_commit=revision,
        checkpoint_sha256=digest(args.checkpoint),
        weights_sha256=digest(out / "model.safetensors"),
        strict_load_verified=True,
        exact_weight_roundtrip=True,
        exact_cpu_logits_roundtrip=True,
        tied_embeddings_verified=True,
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "requirements.txt").write_text("torch==2.6.0\nsafetensors>=0.4\ntokenizers==0.22.2\n")
    shutil.copyfile(root / "publication/generate_completed.py", out / "generate.py")
    for name in ("scholarly-modern-10m-5b-training-audit-20261018.json",):
        shutil.copyfile(artifact_path(args.artifacts, "results/" + name), out / name)
    print(json.dumps({"output": str(out), "parameters": 9849536, "tokens": saved["tokens"]}))


if __name__ == "__main__":
    main()
