# Vanilla Shakespeare GPT, approximately 10M parameters

The complete, readable architecture is in [model.py](model.py): six pre-LayerNorm
causal decoder blocks, width 384, six attention heads, learned positions, GELU
MLPs, and a separate output projection. With 65 characters and context 256 it has
**10,795,776 parameters**. [`100m/model.py`](../100m/model.py) imports this definition unchanged and only
scales its dimensions; the local UI remains separate.

The model starts from random weights and learns next-character prediction from
Tiny Shakespeare. This is a learning benchmark, not an instruction-following
assistant. A contiguous 90/10 train/validation split prevents windows crossing
the boundary; vocabulary is drawn from the whole corpus. There is no sealed test.

## Bounded Modal training

Run from `training/`, using a Python environment with Modal's API proxy support.
The Modal image explicitly pins PyTorch 2.6.0, safetensors 0.8.0 and tqdm 4.70.1,
independently of the newer version selected by the local training project's lock.
The executed versions and source hashes are saved in every run's config.

The published runs used the former `modal_shakespeare.py` (commit `ec5ebd1`). The
same settings are now the [`shakespeare-10m` recipe](../recipes/shakespeare-10m.json),
launched with the generic [recipe launcher](../recipes/README.md) (use a new run ID):

```bash
uv run --with 'modal[api-proxy-support]==1.6.1' modal run modal_train.py \
  --recipe recipes/shakespeare-10m.json --run-id shakespeare-10m-smoke-02 --smoke
```

The original runs used app `lm-exploration-shakespeare-10m` and volume
`lm-exploration-training`. The smoke worker has a 120-second hard timeout; the
training worker a 1,500-second hard timeout and a cooperative 1,200-second training
limit. Both use one L4, no retries, and persist artifacts on exit. A separate
ignored `runs/shakespeare-budget.json` (now `runs/modal-budget.json`) reserved the full worker time at a
conservative $5/hour, with a $3 allocation ceiling. This is reservation accounting,
not an actual bill or an account-wide spending limit. No latent-workspace ledger
or research volume is used.

Training targets 4,000 updates with batch 32, context 256, FP32 AdamW,
learning rate 0.0003, betas (0.9, 0.95), weight decay 0.1, clipping norm 1 and seed
1337. Every 200 updates, evaluate 20 fixed-seed validation batches and save a
checkpoint. Select the saved checkpoint with the lowest validation loss.
Cooperative stopping logs/evaluates/saves the final completed update. Each run
writes a report, metrics, source/runtime/dataset provenance and a sample.

## Export and publication

The published model was exported with `publish_shakespeare.py`, removed after
publication (see commit `ec5ebd1`). It exported the validation-selected checkpoint
to safetensors without optimizer or RNG state, checked the parameter count,
selected update and architecture hash, and uploaded only inference files.
The public bundle includes `model.py`, `generate.py`, safetensors weights, exact
configuration/tokenizer, provenance, requirements and the measured training report.
From the downloaded bundle run `python generate.py --prompt 'ROMEO:' --tokens 300`.
The model uses custom PyTorch code, not Transformers AutoModel loading.

## Completed training and public model

[Public Hugging Face model](https://huggingface.co/jfrick/shakespeare-vanilla-gpt-10m). The bounded L4 run completed 4,000 updates
in 503.3 worker seconds, with 32,768,000 training-character
exposures. Validation selected **update 1,800**, after 14,745,600 exposures:
**1.54483 nats/character**, character perplexity **4.687**.
The final update scored 2.10690 on validation; later training overfit this small
corpus, so the public weights come from the earlier checkpoint.

[Full training report](../results/shakespeare-10m-train-01.json),
[loss history](../results/shakespeare-10m-metrics.jsonl),
[smoke report](../results/shakespeare-10m-smoke-01.json),
and [publication receipt](../results/shakespeare-10m-publication.json).
One seed, validation used for checkpoint selection, no untouched test result.

Local verification covered the 10M parameter count, causal attention and gradients
against SDPA, safe stopping/checkpoint saving, reproducible CPU resume, and exact
exported logits/seeded generation against the selected training checkpoint.
Ruff lint/format, lock consistency and Git diff checks passed.

Pinned public release: `48b3eb9b56fd03056943bb905b8cd7637009588c`. Anonymous
Hub download verified the weights and definition checksums, strict model loading
and exact seeded generation from the downloaded standalone package.
