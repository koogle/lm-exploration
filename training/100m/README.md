# Updated 100M-family architecture

A subsequent [bounded subword pilot](PILOT.md) trained this architecture for
4.08M tokens on Modal. See its [measured report](../results/scholarly-124m-pilot-20261016-01.md);
the architecture-merge verification described below remains separate.

`model.py` uses **RMSNorm, RoPE, SwiGLU and tied input/output embeddings** by
default. There is one `100m` model choice, keeping 14 layers, width 768, 12 ordinary
attention heads (head dimension 64) and context 256. The original vanilla 10M
model and its individual ablation PRs are unchanged.

| Component | Current default |
| --- | --- |
| Normalization | Pre-RMSNorm and final RMSNorm, epsilon 1e-5 |
| Positions | RoPE on Q/K in every layer, theta 10,000 |
| Feedforward | SwiGLU, hidden width 2,048 |
| Projection biases | Present |
| Attention kernel | Original explicit causal MHA |
| Initialization | Normal std 0.02 |
| Input/output weights | One shared Parameter |

## Mechanisms and references

- **RMSNorm:** `gamma * x / sqrt(mean(x**2) + epsilon)` omits mean centering and
  normalization offsets. Epsilon matches historical LayerNorm; low-precision
  statistics accumulate in FP32. [Zhang & Sennrich (2019)](https://arxiv.org/abs/1910.07467)
  motivate simpler normalization. Actual speed depends on kernels.
- **RoPE:** rotates adjacent query/key feature pairs; values stay unchanged.
  The positional dependence of Q/K dot products is relative.
  [RoFormer, Su et al. (2021)](https://arxiv.org/abs/2104.09864) gives the construction.
  Context remains capped by `block_size`: no GQA, KV cache or long-context scaling.
- **SwiGLU:** `down(SiLU(gate(x)) * up(x))` provides input-dependent feature
  modulation. Hidden width `ceil(8 * width / 3)` exactly matches the old MLP's
  matrix-parameter budget at width 768. Projections keep biases.
  [Shazeer (2020)](https://arxiv.org/abs/2002.05202) reports matched-budget T5 gains;
  that motivates a causal-GPT experiment, not a claimed improvement here.
- **Tied embeddings:** `self.lm_head.weight = self.token_embedding.weight`.
  The shared table `E[vocabulary, width]` supplies input rows and output logits
  `hidden @ E.T`. Gradients from both uses accumulate into the same Parameter,
  which appears once in the optimizer's parameter list.
  [Press & Wolf (2017)](https://arxiv.org/abs/1608.05859) study the parameter/quality
  tradeoff. Sharing saves parameters but constrains the representations.

[LLaMA §2.2](https://arxiv.org/abs/2302.13971) adopts RMSNorm, RoPE and SwiGLU.
This update combines the requested choices; it cannot establish each component's
quality contribution. The individual 10M ablations serve that purpose. No quality
gain has been measured for this combined model.

## Frozen tokenizer and parameter count

Use merged `structured-code-v1`: **32,768 entries**, unchanged tokenizer hash
`add5b2c7542e4efded329735825ecdcb94df1432617c3d3e4b7508aa459432f9`.
At context 256 and the unchanged 14×768 shape, the model has **124,389,632 unique
parameters**. Tying saves **25,165,824** (`32,768 × 768`) versus separate matrices
on the same modern backbone. There is no maintained untied modern variant,
separate package or recipe key for tying.

The historical vanilla 100M had 149,760,000 parameters at this vocabulary.
Besides tying, the modern blocks remove 196,608 position parameters and 22,272
normalization offsets, and add 14,336 FFN bias parameters. FFN matrix parameters
match. The `100m` label is historical: a 32K vocabulary changes the total relative
to the character model. No saving is reinvested in extra layers.

## Usage and compatibility

```python
from model_100m.model import GPT, ModelConfig

model = GPT(ModelConfig(vocab_size=32768))
assert model.lm_head.weight is model.token_embedding.weight
```

Recipes continue to select `model="100m"`; no tying flag is needed. New checkpoints
record `architecture_version=2`. A small version-1 compatibility path reuses the
original vanilla implementation to load historical 100M checkpoints, preserving
their original untied weights and exact logits. Missing version metadata means
legacy; the loader does not silently tie or modernize old weights. Resume across
architecture versions is rejected. This is not a second modern model variant.

The general recipe trainer and generation CLI remain character-only; the frozen
BPE preparation pipeline still needs model-training integration. Tests run real
frozen-tokenizer IDs through tiny modern models, not a completed subword
pretraining or text-generation workflow. No GPU run is part of this update.

## Verification

From `training/`:

```sh
uv sync --locked --extra corpus --extra research
uv run --extra corpus --extra research python -m unittest discover -s tests -v
```

Checks cover independent block output/gradient references, causality, finite
gradients, counts, config bounds and old checkpoint loading/resume. Tying checks
cover shared identity, summed gradients against a test-only reference, optimizer
deduplication, dtype conversion, checkpoint roundtrip and tokenizer ID bounds.
