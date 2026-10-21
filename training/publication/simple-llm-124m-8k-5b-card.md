---
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

This is the **completed 5B-token experiment checkpoint**, trained from random initialization.
Completed training: **4,999,981,176 nonpadding next-token exposures / 64,227 optimizer updates**.
The target is 5,000,000,000 exposures; a complete global batch may finish slightly below it.
This is not instruction-tuned, and no downstream capability or full-8K effectiveness is claimed.

## Completed run and training curves

The run finished on 2026-10-18 UTC after **64,227 optimizer updates** and
**4,999,981,176 nonpadding next-token exposures**, 18,824 below the nominal 5B
ceiling because another full global batch would exceed it. Recorded training-worker
session time sums to **7.67 hours**; preparation and queueing are excluded.
The final mixture-weighted diagnostic validation loss is **1.8708 nats/token**.
This diagnostic uses one fixed window per source. It is not a broad evaluation,
benchmark, or demonstration of effective reasoning over the entire 8K context.

![Training loss, validation loss, learning rate, gradient norm, throughput and valid-target fraction](training-curves.png)

The plots include a 200-update moving average for noisy training quantities.
Throughput counts nonpadding targets and excludes validation/checkpoint pauses.
Raw metrics are in `training-metrics.csv`, diagnostics in `validation-diagnostics.json`,
and completion checks in `completed-training.json`. All 64,227 metric updates are
consecutive and finite; exposure totals, fresh-job diagnostic validation continuity,
and the per-source two-pass limit were checked.

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

Source: https://github.com/koogle/lm-exploration . Run: `simple-llm-124m-8k-5b-20261018`.
Full long-context quality evaluation is pending.
