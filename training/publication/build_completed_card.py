"""Build the completed model card from measured validation and training reports."""

import argparse
import json
import math
import shutil
from pathlib import Path

from publication.artifacts import artifact_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    audit_path = artifact_path(
        args.artifacts, "results/scholarly-modern-10m-5b-training-audit-20261018.json"
    )
    validation_path = artifact_path(
        args.artifacts, "results/scholarly-modern-10m-5b-validation-20261018.json"
    )
    audit = json.loads(audit_path.read_text())
    validation = json.loads(validation_path.read_text())
    assert validation["strict_load"] and validation["per_source_two_pass_limit_verified"]
    trained = validation["trained"]
    random = validation["random_initialization_seed42"]
    tokens = sum(r["tokens"] for r in trained["per_source"].values())
    rows = "\n".join(
        f"| {name} | {r['tokens']:,} | {r['loss']:.4f} | {r['perplexity']:.2f} |"
        for name, r in trained["per_source"].items()
    )
    sample_text = "\n\n".join(
        "Prompt: " + s["prompt"] + "\n\n```text\n" + s["completion"] + "\n```"
        for s in validation["samples"][:2]
    )
    card = f"""---
language: en
library_name: pytorch
tags:
- causal-lm
- from-scratch
- custom-code
- experimental
- base-model
---
# Scholarly/code modern GPT — 9.85M parameters, 5B-token experiment

A **base language model trained from random initialization**, with **9,849,536 parameters**, **4,999,934,057 next-token training exposures** and **72,805 optimizer updates**. It is not instruction-tuned. Generated samples contain repetition, invented terminology and factual errors; the model is a small research artifact rather than a reliable assistant.

## Architecture and tokenizer

Eight decoder layers, width 192, three attention heads (64 dimensions/head), RMSNorm, RoPE (theta 10,000), SwiGLU (hidden width 512), causal SDPA with zero dropout, and tied input/output embeddings. Context: 2,048. The frozen `structured-code-v1` byte-level BPE vocabulary has 32,768 entries. PAD=0, BOS=1, EOS=2, UNK=3; ignored training labels use -100, which is not a vocabulary ID. Embeddings account for 6,291,456 parameters; the rest of the model has 3,558,080.

## How it was trained

Three NVIDIA H100 80GB GPUs on Modal using single-host DDP/NCCL, PyTorch 2.6.0+cu124, BF16 autocast with FP32 parameters, fused AdamW (betas 0.9/0.95, weight decay 0.1), gradient clipping at 1.0, and seed 42. Each optimizer update uses 48 document-local windows, microbatch eight per GPU with two accumulation rounds. Padding is excluded from loss and gradients are normalized over actual global targets. Learning rate warms up over 20M tokens to 3e-4, then follows cosine decay to 10% of peak.

The final update stays 65,943 tokens below the nominal 5B ceiling because another complete global batch would exceed it. Model/optimizer, per-rank RNG and consumed data cursors were saved together at optimizer boundaries. The first two bounded sessions tested startup and fresh-job resume before the long segment. All 72,805 recorded steps are consecutive, all recorded losses/gradient norms/learning rates are finite, and optimizer counters/finite state and resume validation boundaries were verified.

The sum of recorded training-worker elapsed time was **{audit["total_session_seconds"] / 3600:.2f} hours** (about {audit["tokens"] / audit["total_session_seconds"]:,.0f} loss tokens/second); upstream preparation, GPU queueing and model evaluation are excluded. This completed run used the original synchronous raw-text loader. The proposed producer/consumer overlap implementation is a separate experiment and did not produce these weights.

## Data

Token-weighted target shares, matched to three decimal places in the run: FineWeb-Edu 10%, permissively licensed Python code 30%, FineMath 20%, peS2o 10%, PubMed 3%, arXiv 2%, Pressbooks 3%, LibreTexts 3%, DOAB 4%, Gutenberg 5%, pre-1929 books 7%, and Library of Congress 3%. The prepared corpus contains 2,745,331,556 eligible training tokens under this tokenizer; the 5B count is exposures, with **at most two passes per source**, verified from checkpoint counters.

Pinned upstream revisions, counts and data hashes are in `data-manifest.json`. The split policy is `scholarly-parent-v2`, grouping books/papers/repositories before windows. Exact duplicate contents are excluded and detected split conflicts quarantined during staging. Global near deduplication, cross-source parent reconciliation, tokenizer fitting-split reconciliation, eligibility and contamination audits remain pending. Therefore the evaluations below are exploratory, not a sealed or contamination-audited benchmark.

## Validation

A fresh evaluation used the first **32 distinct validation parents per source** in content-hash order, at most four 2,048-token windows per parent: **384 parents and {tokens:,} nonpadding targets**. This is a fixed bounded sample, not a random population estimate. Final and seed-42 randomly initialized models were evaluated on the **same targets** using {validation["evaluation_device"].upper()} / PyTorch {validation["torch_version"]}. Loss is token-mean cross-entropy within each source, then combined using the training source weights.

| Model | Mixture-weighted loss (nats/token) | exp(weighted loss) |
|---|---:|---:|
| Random initialization | {random["mixture_weighted_loss"]:.4f} | {math.exp(random["mixture_weighted_loss"]):.2f} |
| Final 5B checkpoint | {trained["mixture_weighted_loss"]:.4f} | {math.exp(trained["mixture_weighted_loss"]):.2f} |

| Source | Evaluated targets | Final loss | Perplexity |
|---|---:|---:|---:|
{rows}

These metrics measure next-token prediction, not factual accuracy, instruction following or functional programming ability. No standardized downstream benchmark score is claimed.

The training-time diagnostic below used only one fixed window per source, so its final loss of 2.6051 is distinct from the broader evaluation above.

![Training diagnostic validation](training-curve.png)

## Unfiltered generation examples

Ten fixed prompts were generated with both greedy decoding and temperature 0.8 / top-k 50, at most 128 new tokens. Seeds and all 20 completions, including poor outputs, are retained in `validation.json`. The first prompt's two completions are shown here without selecting a preferred seed:

{sample_text}

## Download and run

This is standalone custom PyTorch code, not a Transformers AutoModel package. Download the repository, then:

```bash
pip install -r requirements.txt
python generate.py --prompt 'The solar system consists of' --tokens 128 --seed 1000
python generate.py --prompt 'def add(a, b):' --temperature 0 --tokens 64
```

Optional `--device cuda` or `--device mps` uses supported local hardware. Weights are FP32 safetensors. Both tied state-dict keys are stored for strict loading; constructing the supplied model restores shared storage. Optimizer/RNG state remains in the original trusted Modal checkpoint; this bundle alone is not an exact training-resume checkpoint.

Files include the exact model code/dependency, configuration, frozen tokenizer, measured validation and all samples, training audit/curve, and SHA256 provenance. Export verified strict loading, finite parameters, exact safetensors round-trip weights and CPU logits, and tied-embedding storage. Source: https://github.com/koogle/lm-exploration . Run: `scholarly-modern-10m-5b-20261017`; final checkpoint: `scholarly-modern-10m-5b-20261017-s02/step-000072805.pt`.
"""
    (output / "README.md").write_text(card)
    shutil.copyfile(validation_path, output / "validation.json")
    shutil.copyfile(audit_path, output / "training-audit.json")
    for i in range(3):
        shutil.copyfile(
            artifact_path(
                args.artifacts, f"results/scholarly-modern-10m-5b-20261017-s{i:02d}.json"
            ),
            output / f"training-segment-{i:02d}.json",
        )
    print({"evaluation_targets": tokens, "final_loss": trained["mixture_weighted_loss"]})


if __name__ == "__main__":
    main()
