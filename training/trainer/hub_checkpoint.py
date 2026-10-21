"""Export inference checkpoints without disclosing optimizer or RNG state."""

import json
import shutil
from pathlib import Path


def export_checkpoint(checkpoint, output, source, run_id):
    import torch
    from model_100m.model import GPT, ModelConfig
    from safetensors.torch import load_file, save_file
    from training_data.subword import digest

    checkpoint, output, source = Path(checkpoint), Path(output), Path(source)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != "scholarly-ddp-v1":
        raise ValueError("Expected a trusted distributed checkpoint")
    settings = saved["config"]
    cfg = {k: settings[k] for k in ("n_layer", "n_head", "n_embd")}
    cfg.update(vocab_size=32768, block_size=settings["context"], architecture_version=2)
    torch.set_num_threads(4)
    model = GPT(ModelConfig(**cfg)).eval()
    model.load_state_dict(saved["model"], strict=True)
    if sum(p.numel() for p in model.parameters()) != saved["parameters"]:
        raise ValueError("Parameter count mismatch")
    if not all(torch.isfinite(t).all() for t in saved["model"].values()):
        raise ValueError("Non-finite model checkpoint")
    if digest(source / "source/model_100m/model.py") != saved["source_sha256"]["model"]:
        raise ValueError("Export source does not match trained model")
    if sum(saved["data"]["loss_token_exposures"].values()) != saved["tokens"]:
        raise ValueError("Exposure counters disagree")
    output.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous().clone() for k, v in saved["model"].items()},
        str(output / "model.safetensors"),
    )
    restored = load_file(str(output / "model.safetensors"))
    if set(restored) != set(saved["model"]) or any(
        not torch.equal(restored[key], value) for key, value in saved["model"].items()
    ):
        raise ValueError("Safetensors round-trip changed checkpoint weights")
    probe = torch.tensor([[1, 100, 200]])
    with torch.inference_mode():
        expected = model(probe).clone()
        model.load_state_dict(restored, strict=True)
        actual = model(probe)
    if not torch.equal(expected, actual) or not torch.isfinite(actual).all():
        raise ValueError("Exported model logits changed")
    del restored
    for package in ("model_100m", "model_10m"):
        shutil.copytree(
            source / "source" / package,
            output / package,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    shutil.copytree(source / "tokenizer", output / "tokenizer", dirs_exist_ok=True)
    shutil.copyfile(source / "data-manifest.json", output / "data-manifest.json")
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "publication/generate_checkpoint_archive.py",
        output / "generate.py",
    )
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    provenance = {k: v for k, v in saved.items() if k not in ("model", "optimizer", "rng", "data")}
    provenance.update(
        run_id=run_id,
        checkpoint=str(checkpoint),
        weights_sha256=digest(output / "model.safetensors"),
        weight_roundtrip_exact=True,
        cpu_logits_roundtrip_exact=True,
        loss_token_exposures=saved["data"]["loss_token_exposures"],
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "requirements.txt").write_text("torch==2.6.0\nsafetensors>=0.4\ntokenizers==0.22.2\n")
    (output / "README.md").write_text(f"""---
language: en
library_name: pytorch
model_name: Simple LLM 124M
tags:
- causal-lm
- from-scratch
- custom-code
- experimental
---
# Simple LLM 124M — 8K context, 5B-token experiment

This is an **intermediate base-model checkpoint**, trained from random initialization.
Current progress: **{saved["tokens"]:,} nonpadding next-token exposures / {saved["step"]:,} optimizer updates**.
The target is 5,000,000,000 exposures; a complete global batch may finish slightly below it.
This is not instruction-tuned, and no downstream capability or full-8K effectiveness is claimed.

## Training

124,389,632 parameters, 14 layers, width 768, 12 heads, RMSNorm, RoPE (theta 10,000),
SwiGLU and tied 32,768-token embeddings. Configured context: **8,192 tokens**.
Eight H100 80GB GPUs on one Modal host, NCCL/DDP, PyTorch 2.6, BF16 autocast with
FP32 parameters, fused AdamW (betas 0.9/0.95, weight decay 0.1), clipping 1.0,
seed 42. Global batch: 32 document-local windows; microbatch one per GPU with four
accumulation rounds. Padding is ignored; gradients use actual global target counts.
Learning rate: 20M-token warmup to 3e-4, cosine decay to 10% by 5B tokens.
Short documents are padded; configured 8K context does not imply 8K valid tokens per example.

## Data and limitations

The existing frozen corpus is reused, containing 2,745,331,556 eligible unique training
tokens, with at most two passes per source. Target mixture: FineWeb-Edu 10%, Python
30%, FineMath 20%, peS2o 10%, PubMed 3%, arXiv 2%, and books 25% (Pressbooks 3%,
LibreTexts 3%, DOAB 4%, Gutenberg 5%, pre-1929 books 7%, Library of Congress 3%).
Raw documents are selected by token-weighted deficits, tokenized just in time and
prefetched with only consumed batches acknowledged in checkpoints.
Pinned revisions and pending audits are preserved in `data-manifest.json`.
Global near-deduplication, cross-source parent reconciliation, tokenizer split
reconciliation, eligibility and contamination review remain pending. This is an
explicitly authorized exploratory run, not an audited production corpus.

## Checkpoints and use

Each publication commit replaces the current inference bundle. Earlier checkpoints
remain available through Hugging Face commit revisions; `provenance.json` records
progress and hashes. Optimizer, per-rank RNG and exact data cursors stay on the Modal
run volume for training resume. This download alone is not an exact resume checkpoint.
The original two-step smoke artifact is preserved on branch `smoke-step2-before-8k`.

Custom PyTorch code, not Transformers AutoModel:

```bash
pip install -r requirements.txt
python generate.py --prompt 'def add(a, b):' --tokens 64
```

Source: https://github.com/koogle/lm-exploration . Run: `{run_id}`.
Full long-context quality evaluation is pending.
""")
    return {"step": saved["step"], "tokens": saved["tokens"], "checkpoint": str(checkpoint)}
