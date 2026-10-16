# Updated 100M-family architecture

`model.py` now defaults to **RMSNorm, RoPE and SwiGLU**, keeping 14 layers,
width 768, 12 ordinary attention heads (head dimension 64) and context 256.
The [tied variant](../100m_tied/README.md) inherits these blocks and shares the
input/output embedding Parameter. The published vanilla 10M is unchanged.

| Component | Historical version 1 | Default version 2 |
| --- | --- | --- |
| Normalization | Pre-LayerNorm + final LayerNorm | Pre-RMSNorm + final RMSNorm |
| Positions | Learned absolute table at input | RoPE on Q/K in every layer |
| Feedforward | GELU, hidden width 3,072 | SwiGLU, hidden width 2,048 |
| Projection biases | Present | Present |
| Attention kernel | Explicit causal MHA | Same explicit causal MHA |
| Initialization | Normal std 0.02 | Same normal std 0.02 |
| `100m` embedding/output weights | Separate | Separate |
| `100m_tied` embedding/output weights | Shared | Shared |

## Mechanisms and evidence

- **RMSNorm:** `gamma * x / sqrt(mean(x**2) + epsilon)` omits mean centering
  and normalization offsets. Epsilon remains **1e-5**, matching old LayerNorm.
  Low-precision statistics accumulate in FP32. [Zhang & Sennrich (2019)](https://arxiv.org/abs/1910.07467)
  motivate simpler magnitude normalization. Actual speed depends on kernels;
  fewer operations do not establish a measured speedup here.
- **RoPE:** rotates adjacent Q/K feature pairs using theta **10,000**. Values
  stay unchanged; the positional dependence of Q/K dot products is relative.
  [RoFormer, Su et al. (2021)](https://arxiv.org/abs/2104.09864) gives the construction.
  Context is still bounded by `block_size`: no GQA, KV cache or long-context
  scaling is added.
- **SwiGLU:** `down(SiLU(gate(x)) * up(x))` provides input-dependent feature
  modulation. Hidden width `ceil(8 * width / 3)` exactly matches the old MLP's
  matrix-parameter count at width 768. All three projections retain biases.
  [Shazeer (2020)](https://arxiv.org/abs/2002.05202) reports matched-budget T5
  improvements over GELU; that motivates a causal-GPT experiment, not a claimed
  improvement in this repository.

[LLaMA §2.2](https://arxiv.org/abs/2302.13971) adopts all three in a decoder.
This update combines them as requested; it cannot establish each component's
quality contribution. The independent 10M ablations serve that purpose.

## Counts with the merged frozen tokenizer

Use existing `structured-code-v1`: **32,768 entries**, unchanged tokenizer hash
`add5b2c7542e4efded329735825ecdcb94df1432617c3d3e4b7508aa459432f9`.
At context 256 and the unchanged 14×768 shape:

| Version | Untied `100m` | Tied `100m_tied` |
| --- | ---: | ---: |
| 1: historical blocks | 149,760,000 | 124,594,176 |
| 2: RMSNorm + RoPE + SwiGLU | **149,555,456** | **124,389,632** |

Version 2 removes 196,608 position parameters and 22,272 normalization offsets,
and adds 14,336 FFN bias parameters. FFN matrix parameters match. Tying saves
**25,165,824** parameters in either version. The `100m` name is historical:
including a 32K vocabulary changes its total relative to the character model.

## Versioning and integration

```python
from model_100m.model import GPT, ModelConfig

modern = ModelConfig(vocab_size=32768)  # architecture_version=2
legacy = ModelConfig(vocab_size=32768, architecture_version=1)
```

Recipes selecting `100m` or `100m_tied` build version 2 by default. For a legacy
control, use `model_overrides: {"architecture_version": 1}`. New checkpoints
record the version. The loader interprets pre-version 100M/tied-100M checkpoints
as version 1, preserving their original blocks and exact logits; it does not
silently convert weights. Resume across versions or tied/untied architectures
is rejected. Explicit version 1 can resume a compatible pre-version checkpoint.

The general recipe trainer and generation CLI remain character-only; the frozen
BPE preparation pipeline still needs a model-training integration. Tests run
actual frozen-tokenizer IDs through tiny modern models, not a completed subword
pretraining or text-generation workflow. No GPU run or comparative quality
result is part of this update.

## Verification

From `training/`:

```sh
uv sync --locked --extra corpus --extra research
uv run --extra corpus --extra research python -m unittest discover -s tests -v
```

Checks cover independent RMSNorm/RoPE/SwiGLU output/gradient references, combined
causality and finite gradients, context/config bounds, counts, recipe versioning,
and old checkpoint loading/resume. The tying tests cover shared identity, summed
input/output gradients, optimizer deduplication, dtype conversion, checkpoint
roundtrip and the merged tokenizer's full ID range.
