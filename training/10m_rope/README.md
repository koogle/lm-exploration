# 10M RoPE-only experiment

This variant replaces learned absolute positional embeddings with rotary
query/key embeddings in every attention layer. It preserves the [vanilla 10M](../10m/README.md)
shape: 6 layers, width 384, 6 MHA heads, head dimension 64, context 256 and
character tokens. Pre-LayerNorm, GELU, matrix biases, untied input/output
embeddings, explicit causal attention and normal initialization with standard
deviation 0.02 stay unchanged. No RMSNorm, SwiGLU, GQA or kernel change is included.

## Mechanism and evidence

For position `p`, each adjacent feature pair rotates by
`p * 10000**(-2*j/head_dim)`, where `j` indexes the pair. Queries and keys rotate;
values do not. Rotation preserves magnitude, and a query/key dot product depends
on the difference between their positions. No learned position table is added
to the input embedding.

[Su et al., RoFormer (2021)](https://arxiv.org/abs/2104.09864) derives rotary
positions and analyzes the relative-position dependence.
[LLaMA §2.2](https://arxiv.org/abs/2302.13971) adopts RoPE for a causal decoder.
The hypothesis is useful positional generalization, not guaranteed better loss
or useful extrapolation to unseen lengths. First compare held-out loss at the
same trained context. Context remains capped at 256 in this recipe; this change
does not implement long-context scaling or a KV cache.

At vocabulary 65 and context 256:

| Architecture | Parameters | Difference |
| --- | ---: | ---: |
| Vanilla | 10,795,776 | — |
| RoPE only | 10,697,472 | −98,304 |

The difference is the removed `256 × 384` position table. Do not spend that
parameter saving on wider layers in this comparison. The initialization
distribution is preserved; the changed module construction means a shared seed
alone does not guarantee identical values for every unchanged tensor.

## Verify and compare

From `training/`:

```sh
uv sync --locked
uv run python -m unittest discover -s tests -p test_10m_rope.py -v
uv run ruff check 10m_rope tests/test_10m_rope.py trainer/recipe.py trainer/loop.py
uv run python -c 'from trainer.recipe import load_recipe; print(load_recipe("recipes/shakespeare-10m-rope.json").model)'
```

Checks compare rotation with an independent complex-number reference, verify
norm preservation and invariance to a common position shift, check numerical
gradients, compare attention outputs/gradients with reference causal attention,
and cover causality, sequence bounds, odd-head rejection, parameter counts,
recipe parity, architecture-aware checkpoint loading and source provenance.
These are bounded CPU correctness checks, not a local training entrypoint.

The recipe [`shakespeare-10m-rope.json`](../recipes/shakespeare-10m-rope.json)
matches the vanilla data, tokenizer, training hyperparameters, GPU and limits;
only its model selection/description differ. Before paid runs restore the
allocation ledger and use fresh run IDs:

```sh
uv run --with 'modal[api-proxy-support]==1.6.1' modal run modal_train.py \
  --recipe recipes/shakespeare-10m-rope.json --run-id rope-10m-smoke-01 --smoke
uv run --with 'modal[api-proxy-support]==1.6.1' modal run modal_train.py \
  --recipe recipes/shakespeare-10m-rope.json --run-id rope-10m-train-01
```

Rerun `shakespeare-10m.json` as the control under the same runtime and token
budget; compare validation loss at matched token exposure and elapsed time,
using the same checkpoint-selection policy. Recheck promising differences
across paired seeds before scaling. A wall-time-limited run with fewer updates
is not an equal-token comparison. No GPU training or quality comparison has been
executed for this variant.

The vanilla and 100M definitions are preserved. `BASE_MODEL_SOURCE` lets the
trainer hash both the variant and inherited vanilla source. New checkpoints
record `model_10m_rope`; `lm-generate` already dispatches through the model
registry. Start this architecture from scratch, not by resuming vanilla weights.
