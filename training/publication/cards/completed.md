+++
audit = "results/scholarly-modern-10m-5b-training-audit-20261018.json"
validation = "results/scholarly-modern-10m-5b-validation-20261018.json"
segments = [
  "results/scholarly-modern-10m-5b-20261017-s00.json",
  "results/scholarly-modern-10m-5b-20261017-s01.json",
  "results/scholarly-modern-10m-5b-20261017-s02.json",
]
baseline = "random_initialization_seed42"
baseline_label = "Random initialization (seed 42)"
displayed_samples = 2

# These settings were fixed in the historical trainer and omitted from its JSON.
# A report that records them takes precedence over this historical documentation.
[historical_training]
adam_beta1 = 0.9
adam_beta2 = 0.95
gradient_clip_norm = 1.0
validation_windows_per_source = 1
+++
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
# Simple LLM 10M — ${parameters} parameters, ${target_tokens}-token experiment

A **base language model trained from random initialization**, with **${parameters} parameters**, **${trained_tokens} next-token training exposures** and **${updates} optimizer updates**. It is not instruction-tuned. Generated samples contain repetition, invented terminology and factual errors; the model is a small research artifact rather than a reliable assistant.

## Architecture and tokenizer

${layers} decoder layers, width ${width}, ${heads} attention heads (${head_dimension} dimensions/head), RMSNorm, RoPE (theta 10,000), SwiGLU, causal SDPA with zero dropout, and tied input/output embeddings. Context: ${context}. The frozen `structured-code-v1` byte-level BPE vocabulary has 32,768 entries. PAD=0, BOS=1, EOS=2, UNK=3; ignored training labels use -100, which is not a vocabulary ID.

## How it was trained

${world_size} ${gpu} GPUs on Modal using single-host DDP/NCCL, PyTorch ${training_torch_version}, BF16 autocast with FP32 parameters, fused AdamW (betas ${adam_beta1}/${adam_beta2}, weight decay ${weight_decay}), gradient clipping at ${gradient_clip_norm}, and seed ${seed}. Each optimizer update uses ${global_windows} document-local windows, microbatch ${microbatch} per GPU with ${accumulation_rounds} accumulation rounds. Padding is excluded from loss and gradients are normalized over actual global targets. Learning rate warms up over ${warmup_tokens} tokens to ${peak_lr}, then follows cosine decay to ${min_lr_percent}% of peak.

The final update stays ${unused_tokens} tokens below the nominal ${target_tokens}-token ceiling because another complete global batch would exceed it. Model/optimizer, per-rank RNG and consumed data cursors were saved together at optimizer boundaries. The first two bounded sessions tested startup and fresh-job resume before the long segment. All ${updates} recorded steps are consecutive, all recorded losses/gradient norms/learning rates are finite, and optimizer counters/finite state and resume validation boundaries were verified.

The sum of recorded training-worker elapsed time was **${session_hours} hours** (about ${tokens_per_second} loss tokens/second); upstream preparation, GPU queueing and model evaluation are excluded. This completed run used the original synchronous raw-text loader. The proposed producer/consumer overlap implementation is a separate experiment and did not produce these weights.

## Data

Actual token-weighted training exposure shares: ${source_shares}. The prepared corpus contains ${unique_train_tokens} eligible training tokens under this tokenizer; the training-token count is exposures, with **at most two passes per source**, verified from checkpoint counters.

Pinned upstream revisions, counts and data hashes are in `data-manifest.json`. The split policy is `scholarly-parent-v2`, grouping books/papers/repositories before windows. Exact duplicate contents are excluded and detected split conflicts quarantined during staging. Pending audits recorded in the evaluation: ${pending_audits}. Therefore the evaluations below are exploratory, not a sealed or contamination-audited benchmark.

## Validation

${evaluation_policy}

The evaluated sample contains **${evaluation_parents} parents and ${evaluation_targets} nonpadding targets**. Final and randomly initialized models were evaluated on the **same targets** using ${evaluation_device} / PyTorch ${evaluation_torch_version}. Loss is token-mean cross-entropy within each source, then combined using the training source weights.

| Model | Mixture-weighted loss (nats/token) | exp(weighted loss) |
|---|---:|---:|
| ${baseline_label} | ${baseline_loss} | ${baseline_perplexity} |
| Final checkpoint | ${trained_loss} | ${trained_perplexity} |

| Source | Evaluated targets | Final loss | Perplexity |
|---|---:|---:|---:|
${source_rows}

These metrics measure next-token prediction, not factual accuracy, instruction following or functional programming ability. No standardized downstream benchmark score is claimed.

Training-time validation windows per source: **${validation_windows_per_source}**. The diagnostic's final loss of ${diagnostic_loss} is distinct from the broader evaluation above.

![Training diagnostic validation](training-curve.png)

## Unfiltered generation examples

${prompt_count} fixed prompts were generated with both greedy decoding and temperature 0.8 / top-k 50, at most 128 new tokens. Seeds and all ${completion_count} completions, including poor outputs, are retained in `validation.json`. The first ${displayed_samples} recorded completions are shown in report order:

${sample_text}

## Download and run

This is standalone custom PyTorch code, not a Transformers AutoModel package. Download the repository, then:

```bash
pip install -r requirements.txt
python generate.py --prompt 'The solar system consists of' --tokens 128 --seed 1000
python generate.py --prompt 'def add(a, b):' --temperature 0 --tokens 64
```

Optional `--device cuda` or `--device mps` uses supported local hardware. Weights are FP32 safetensors. Both tied state-dict keys are stored for strict loading; constructing the supplied model restores shared storage. Optimizer/RNG state remains in the original trusted Modal checkpoint; this bundle alone is not an exact training-resume checkpoint.

Files include the exact model code/dependency, configuration, frozen tokenizer, measured validation and all samples, training audit/curve, and SHA256 provenance. Export verified strict loading, finite parameters, exact safetensors round-trip weights and CPU logits, and tied-embedding storage. Source: https://github.com/koogle/lm-exploration . Training segments: ${segment_count}; final checkpoint: `${final_checkpoint}`.
