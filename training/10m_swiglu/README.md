# 10M SwiGLU-only experiment

This variant replaces each GELU feedforward layer with SwiGLU. It preserves the
[vanilla 10M](../10m/README.md) shape: 6 layers, width 384, 6 MHA heads, learned
positions, context 256 and character tokens. Pre-LayerNorm, all matrix biases,
explicit causal attention, untied input/output embeddings and normal
initialization with standard deviation 0.02 stay unchanged. No normalization or
position-encoding change is included.

## Mechanism and evidence

The original MLP is `down(GELU(up(x)))`, with two matrices and hidden width 1,536.
SwiGLU is `down(SiLU(gate(x)) * up(x))`: an input-dependent gate modulates the
expanded features before projecting back. The three-matrix SwiGLU uses hidden
width **1,024**, preserving the default model's feedforward matrix-parameter count:
`3 * 384 * 1024 = 2 * 384 * 1536`.

[Shazeer, GLU Variants Improve Transformer (2020), Table 1](https://arxiv.org/abs/2002.05202)
reports held-out log-perplexity 1.636 for SwiGLU versus 1.679 for GELU after
524,288 steps in a matched-parameter/compute T5 experiment.
[LLaMA §2.2](https://arxiv.org/abs/2302.13971) adopts SwiGLU with reduced hidden
width. That T5 result motivates an experiment, not a claim about this causal
10M model. Measure held-out loss at matched token exposure, plus throughput and
memory: another projection/gating operation can change actual runtime.

Unlike the bias-free FFNs studied in the GLU paper, this ablation keeps biases
on all three projections to preserve vanilla's bias policy. This introduces a
small, explicit bias-parameter difference:

| Architecture | Parameters | Difference |
| --- | ---: | ---: |
| Vanilla | 10,795,776 | — |
| SwiGLU only | 10,798,848 | +3,072 |

The figures assume vocabulary 65 and context 256. Matrix parameters match
exactly; bias parameters increase by 512 per block. For other test widths,
`ceil(8 * width / 3)` ensures a positive hidden width and reports its rounding
through the actual parameter count. This is not an exact reproduction of the
paper's entire model or training recipe.

## Verify and compare

From `training/`:

```sh
uv sync --locked
uv run python -m unittest discover -s tests -p test_10m_swiglu.py -v
uv run ruff check 10m_swiglu tests/test_10m_swiglu.py trainer/recipe.py trainer/loop.py
uv run python -c 'from trainer.recipe import load_recipe; print(load_recipe("recipes/shakespeare-10m-swiglu.json").model)'
```

Checks compare output/input/parameter gradients with an explicit Swish-gate
reference, verify matrix budget and biases, exact preservation of non-MLP weights
under the same seed, causality, finite whole-model gradients, parameter counts,
recipe parity, architecture-aware checkpoint loading and source provenance.
These are bounded CPU correctness checks, not a local training entrypoint.

The recipe [`shakespeare-10m-swiglu.json`](../recipes/shakespeare-10m-swiglu.json)
matches the vanilla data, tokenizer, training hyperparameters, GPU and limits;
only its model selection/description differ. Before paid runs restore the
allocation ledger and use fresh run IDs:

```sh
uv run --with 'modal[api-proxy-support]==1.6.1' modal run modal_train.py \
  --recipe recipes/shakespeare-10m-swiglu.json --run-id swiglu-10m-smoke-01 --smoke
uv run --with 'modal[api-proxy-support]==1.6.1' modal run modal_train.py \
  --recipe recipes/shakespeare-10m-swiglu.json --run-id swiglu-10m-train-01
```

Rerun `shakespeare-10m.json` as the control under the same runtime and token
budget; compare validation loss at matched token exposure and elapsed time,
using the same checkpoint-selection policy. Recheck promising differences
across paired seeds before scaling. A wall-time-limited run with fewer updates
is not an equal-token comparison. No GPU training or quality comparison has been
executed for this variant.

The vanilla and 100M definitions are preserved. `BASE_MODEL_SOURCE` lets the
trainer hash both the variant and inherited vanilla source. New checkpoints
record `model_10m_swiglu`; `lm-generate` already dispatches through the model
registry. Start this architecture from scratch, not by resuming vanilla weights.
