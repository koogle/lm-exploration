---
language: en
library_name: pytorch
model_name: Simple LLM 10M
tags:
- causal-lm
- from-scratch
- custom-code
- experimental
- base-model
---
# Simple LLM 10M — 9.85M parameters, 5B-token experiment

A **base language model trained from random initialization**, with **9,849,536 parameters**, **4,999,934,057 next-token training exposures** and **72,805 optimizer updates**. It is not instruction-tuned. Generated samples contain repetition, invented terminology and factual errors; the model is a small research artifact rather than a reliable assistant.

## Architecture and tokenizer

Eight decoder layers, width 192, three attention heads (64 dimensions/head), RMSNorm, RoPE (theta 10,000), SwiGLU (hidden width 512), causal SDPA with zero dropout, and tied input/output embeddings. Context: 2,048. The frozen `structured-code-v1` byte-level BPE vocabulary has 32,768 entries. PAD=0, BOS=1, EOS=2, UNK=3; ignored training labels use -100, which is not a vocabulary ID. Embeddings account for 6,291,456 parameters; the rest of the model has 3,558,080.

## How it was trained

Three NVIDIA H100 80GB GPUs on Modal using single-host DDP/NCCL, PyTorch 2.6.0+cu124, BF16 autocast with FP32 parameters, fused AdamW (betas 0.9/0.95, weight decay 0.1), gradient clipping at 1.0, and seed 42. Each optimizer update uses 48 document-local windows, microbatch eight per GPU with two accumulation rounds. Padding is excluded from loss and gradients are normalized over actual global targets. Learning rate warms up over 20M tokens to 3e-4, then follows cosine decay to 10% of peak.

The final update stays 65,943 tokens below the nominal 5B ceiling because another complete global batch would exceed it. Model/optimizer, per-rank RNG and consumed data cursors were saved together at optimizer boundaries. The first two bounded sessions tested startup and fresh-job resume before the long segment. All 72,805 recorded steps are consecutive, all recorded losses/gradient norms/learning rates are finite, and optimizer counters/finite state and resume validation boundaries were verified.

The sum of recorded training-worker elapsed time was **3.11 hours** (about 447,114 loss tokens/second); upstream preparation, GPU queueing and model evaluation are excluded. This completed run used the original synchronous raw-text loader. The proposed producer/consumer overlap implementation is a separate experiment and did not produce these weights.

## Data

Token-weighted target shares, matched to three decimal places in the run: FineWeb-Edu 10%, permissively licensed Python code 30%, FineMath 20%, peS2o 10%, PubMed 3%, arXiv 2%, Pressbooks 3%, LibreTexts 3%, DOAB 4%, Gutenberg 5%, pre-1929 books 7%, and Library of Congress 3%. The prepared corpus contains 2,745,331,556 eligible training tokens under this tokenizer; the 5B count is exposures, with **at most two passes per source**, verified from checkpoint counters.

Pinned upstream revisions, counts and data hashes are in `data-manifest.json`. The split policy is `scholarly-parent-v2`, grouping books/papers/repositories before windows. Exact duplicate contents are excluded and detected split conflicts quarantined during staging. Global near deduplication, cross-source parent reconciliation, tokenizer fitting-split reconciliation, eligibility and contamination audits remain pending. Therefore the evaluations below are exploratory, not a sealed or contamination-audited benchmark.

## Validation

A fresh evaluation used the first **32 distinct validation parents per source** in content-hash order, at most four 2,048-token windows per parent: **384 parents and 1,798,432 nonpadding targets**. This is a fixed bounded sample, not a random population estimate. Final and seed-42 randomly initialized models were evaluated on the **same targets** using CPU / PyTorch 2.6.0+cpu. Loss is token-mean cross-entropy within each source, then combined using the training source weights.

| Model | Mixture-weighted loss (nats/token) | exp(weighted loss) |
|---|---:|---:|
| Random initialization | 10.4369 | 34096.37 |
| Final 5B checkpoint | 2.6926 | 14.77 |

| Source | Evaluated targets | Final loss | Perplexity |
|---|---:|---:|---:|
| fineweb_edu | 33,310 | 3.4005 | 29.98 |
| code_python | 58,082 | 2.1360 | 8.47 |
| finemath_4plus | 53,704 | 2.3665 | 10.66 |
| pes2o | 197,043 | 2.8099 | 16.61 |
| pubmed | 228,568 | 2.7080 | 15.00 |
| arxiv | 233,891 | 2.5063 | 12.26 |
| pressbooks | 46,850 | 3.3014 | 27.15 |
| libretexts | 65,755 | 2.8839 | 17.88 |
| doab | 139,270 | 2.9186 | 18.51 |
| gutenberg | 251,773 | 3.4519 | 31.56 |
| pre_1929_books | 252,531 | 3.5251 | 33.96 |
| library_of_congress | 237,655 | 3.4807 | 32.48 |

These metrics measure next-token prediction, not factual accuracy, instruction following or functional programming ability. No standardized downstream benchmark score is claimed.

The training-time diagnostic below used only one fixed window per source, so its final loss of 2.6051 is distinct from the broader evaluation above.

![Training diagnostic validation](training-curve.png)

## Unfiltered generation examples

Ten fixed prompts were generated with both greedy decoding and temperature 0.8 / top-k 50, at most 128 new tokens. Seeds and all 20 completions, including poor outputs, are retained in `validation.json`. The first prompt's two completions are shown here without selecting a preferred seed:

Prompt: The solar system consists of

```text
 a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 2D system with a 
```

Prompt: The solar system consists of

```text
 relatively poor plastic muscles and humans. In addition, muscles are used as prominent vascular plastic muscles to avoid the muscles and muscles. Very few humans can have a number of bundles. The most common venous vessels, such as cyanobenzoans, are small amounts of muscles used to induce the britis. For example, muscles can be used in the dietary environment. Consumers in Emontan
```

## Download and run

This is standalone custom PyTorch code, not a Transformers AutoModel package. Download the repository, then:

```bash
pip install -r requirements.txt
python generate.py --prompt 'The solar system consists of' --tokens 128 --seed 1000
python generate.py --prompt 'def add(a, b):' --temperature 0 --tokens 64
```

Optional `--device cuda` or `--device mps` uses supported local hardware. Weights are FP32 safetensors. Both tied state-dict keys are stored for strict loading; constructing the supplied model restores shared storage. Optimizer/RNG state remains in the original trusted Modal checkpoint; this bundle alone is not an exact training-resume checkpoint.

Files include the exact model code/dependency, configuration, frozen tokenizer, measured validation and all samples, training audit/curve, and SHA256 provenance. Export verified strict loading, finite parameters, exact safetensors round-trip weights and CPU logits, and tied-embedding storage. Source: https://github.com/koogle/lm-exploration . Run: `scholarly-modern-10m-5b-20261017`; final checkpoint: `scholarly-modern-10m-5b-20261017-s02/step-000072805.pt`.
