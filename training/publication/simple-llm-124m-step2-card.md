---
language: en
library_name: pytorch
model_name: Simple LLM 124M
tags:
- causal-lm
- from-scratch
- smoke-test
- distributed-training
- custom-code
---
# Simple LLM 124M — two-step smoke checkpoint

This is an **infrastructure smoke-test artifact**, not a useful pretrained or instruction-tuned model. It contains exactly the checkpoint after **two optimizer updates / 121,777 nonpadding next-token training targets**. The later four-step resumed checkpoint is not this upload.

## Architecture

124,389,632 unique parameters; 14 decoder layers, width 768, 12 attention heads (64 dimensions/head), RMSNorm, rotary positions (theta 10,000), SwiGLU (hidden width 2,048), and tied input/output embeddings. Context: 2,048. Causal PyTorch scaled-dot-product attention with zero dropout. Frozen `structured-code-v1` byte-level BPE, vocabulary 32,768; PAD=0, BOS=1, EOS=2, UNK=3. Padding targets use -100 (not a vocabulary ID).

## How this checkpoint was produced

Random initialization, seed 42, on **two NVIDIA H100 80GB GPUs on Modal**, using single-host PyTorch DDP/NCCL. PyTorch 2.6.0+cu124, BF16 autocast with FP32 parameters, fused AdamW (betas 0.9/0.95, weight decay 0.1), global gradient clipping at 1.0. Global batch: 48 document-local windows, microbatch 24 per GPU. Padded targets are excluded; gradients are normalized over the global number of actual targets. Peak configured learning rate 3e-4, 100,000-token warmup, followed by cosine decay toward 10% of peak over a configured 5M-token pilot. **Execution stopped at two steps**, well before that pilot target.

The run `scholarly-ddp-full-smoke-20261017-02` wrote `step-000000002.pt` on the `lm-exploration-training` Modal volume. Recorded worker training elapsed time was 39.87 seconds; this tiny run is not a sustained-throughput measurement. A separate fresh job subsequently resumed this checkpoint to step four, exercising the continuation path.

## Data and limitations

Token-weighted source targets: FineWeb-Edu 10%, permissively licensed Python code 30%, FineMath 20%, peS2o 10%, PubMed 3%, arXiv 2%, and books 25% (Pressbooks 3%, LibreTexts 3%, DOAB 4%, Gutenberg 5%, pre-1929 books 7%, Library of Congress 3%). Actual two-step exposures are in `training-report.json`; short-run proportions differ from targets.

This used a **bounded, pending-audit pilot snapshot**, not a qualified production corpus. The split is `scholarly-parent-v2`; reconciliation with tokenizer fitting splits, global near-duplicate removal, cross-source parent reconciliation, eligibility/provenance and benchmark contamination audits remain pending. No generalization or benchmark capability claim is made. Diagnostic validation used one fixed window per source: weighted loss 10.56996 before training and 9.64922 after two updates. These are not robust evaluation scores.

## Files and use

`model.safetensors` contains inference weights; `config.json`, `model_100m/`, and `model_10m/` contain the exact model definition and dependency. `tokenizer/` contains the frozen tokenizer. `training-report.json`, `data-manifest.json`, and `provenance.json` record settings, source hashes, counters and checkpoint identity. Source repository: https://github.com/koogle/lm-exploration . The exporter verifies exact source hash, strict state loading, safetensors round-trip equality, parameter count and finite forward output.

This is custom PyTorch code, **not** a Transformers `AutoModel` package. After downloading the repository:

```bash
pip install -r requirements.txt
python generate.py --prompt 'def add(a, b):' --tokens 32
```

Expect incoherent output after two updates. Optimizer state, data cursor and per-rank RNG remain in the original Modal checkpoint; this inference bundle alone is not an exact training-resume checkpoint. Model weights are stored in FP32, with both tied state-dict keys present; constructing the supplied model restores the tie.
