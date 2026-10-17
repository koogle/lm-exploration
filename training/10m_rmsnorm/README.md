# 10M RMSNorm-only experiment

This variant replaces the two pre-normalizations in each vanilla block and the
final normalization with RMSNorm. It starts from the same 6-layer, width-384,
6-head, context-256 [vanilla 10M](../10m/README.md). Learned positions, GELU,
ordinary multi-head attention, matrix biases, untied embeddings and normal
initialization (standard deviation 0.02) stay unchanged. No other architecture
proposal is included.

## Mechanism and evidence

For a token vector `x` of width `d`, RMSNorm computes
`gamma * x / sqrt(mean(x**2) + epsilon)`. LayerNorm instead subtracts the mean,
divides by `sqrt(mean((x - mean(x))**2) + epsilon)`, and adds a learned offset.
RMSNorm retains magnitude normalization and omits mean centering. It has a
learned scale, but no additive offset. Both use epsilon **1e-5** here to match
the existing vanilla LayerNorm.

[Zhang & Sennrich, Root Mean Square Layer Normalization (2019)](https://arxiv.org/abs/1910.07467)
report comparable quality with lower normalization overhead on their tested
architectures. [LLaMA §2.2](https://arxiv.org/abs/2302.13971) adopts pre-RMSNorm
in a decoder. Neither establishes an improvement in this 10M Shakespeare model.
Fewer arithmetic operations need not beat a fused LayerNorm kernel: measure
held-out loss, sustained throughput, peak memory and time to a fixed loss.

At vocabulary 65 and context 256:

| Architecture | Parameters | Difference |
| --- | ---: | ---: |
| Vanilla | 10,795,776 | — |
| RMSNorm only | 10,790,784 | −4,992 |

The difference is exactly 13 removed normalization offset vectors of width 384.
Input/output matrices remain separate. Low-precision statistics accumulate in
FP32; FP64 is preserved for numerical reference/gradient checks.

## Verify and compare

From `training/`:

```sh
uv sync --locked
uv run python -m unittest discover -s tests -p test_10m_rmsnorm.py -v
uv run ruff check 10m_rmsnorm tests/test_10m_rmsnorm.py trainer/recipe.py trainer/loop.py
uv run python -c 'from trainer.recipe import load_recipe; print(load_recipe("recipes/shakespeare-10m-rmsnorm.json").model)'
```

Checks compare output/input/scale gradients with PyTorch RMSNorm, cover zero and
low-precision inputs, verify exact preservation of non-normalization weights
under the same seed, causality, finite whole-model gradients, parameter counts,
recipe parity, architecture-aware checkpoint loading and source provenance.
These are bounded CPU correctness checks, not a local training entrypoint.

The recipe [`shakespeare-10m-rmsnorm.json`](../recipes/shakespeare-10m-rmsnorm.json)
matches the vanilla recipe's data, tokenizer, training hyperparameters, GPU and
limits; only its model selection/description differ. Before paid runs restore
the allocation ledger and use fresh run IDs:

```sh
uv run --with 'modal[api-proxy-support]==1.6.1' modal run modal_train.py \
  --recipe recipes/shakespeare-10m-rmsnorm.json --run-id rmsnorm-10m-smoke-01 --smoke
uv run --with 'modal[api-proxy-support]==1.6.1' modal run modal_train.py \
  --recipe recipes/shakespeare-10m-rmsnorm.json --run-id rmsnorm-10m-train-01
```

Rerun `shakespeare-10m.json` as the control under the same runtime and token
budget; compare validation loss at matched token exposure and elapsed time,
using the same checkpoint-selection policy. Recheck promising differences
across paired seeds before scaling. A wall-time-limited run with fewer updates
is not an equal-token comparison. No GPU training or quality comparison has been
executed for this variant.

The vanilla and 100M definitions are preserved. `BASE_MODEL_SOURCE` lets the
trainer hash both the variant and inherited vanilla source. New checkpoints
record `model_10m_rmsnorm`; `lm-generate` already dispatches through the model
registry. Start this architecture from scratch, not by resuming vanilla weights.
