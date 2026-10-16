# Tied embeddings for the existing 100M architecture

This is an independent weight-sharing experiment. It inherits the existing
`100m/model.py` configuration and blocks, initializes the same model, then does:

```python
self.lm_head.weight = self.token_embedding.weight
```

Both modules use the **same Parameter**, not equal copies. The embedding table
is `E[vocabulary, width]`: input tokens select rows of `E`; output logits are
`hidden @ E.T`. Gradients from both uses accumulate into that one Parameter,
which appears once in `model.parameters()` and the trainer's AdamW parameter
list. There is no additional initialization of the shared weight.

## Rationale and tradeoff

[Press & Wolf, Using the Output Embedding to Improve Language Models (2017)](https://arxiv.org/abs/1608.05859)
study sharing the input and output representations and report perplexity/size
benefits in their tested models. Sharing removes one vocabulary-by-width matrix
and couples the learned representations. That can be useful regularization,
but it also restricts the model: input lookup and output classification can no
longer learn independent matrices. It is a hypothesis to compare, not a
guaranteed quality improvement in this repository.

Depth, width, context, LayerNorm, learned positions, GELU, attention, biases and
initialization remain unchanged. There is no RoPE, RMSNorm, SwiGLU or GQA change.
The original `100m` and `10m` architectures stay untied. The new registry key is
`100m_tied`; checkpoints identify the package as `model_100m_tied`.

## Actual merged tokenizer and parameter budget

Use the current frozen **structured-code-v1** tokenizer, not the previous 16K
example or a newly fitted tokenizer:

- Vocabulary: **32,768** entries, including the four special tokens.
- Tokenizer SHA-256: `add5b2c7542e4efded329735825ecdcb94df1432617c3d3e4b7508aa459432f9`.
- Loader: `training_data.subword.load_frozen()` verifies the checked-in artifacts.

At the current configuration (14 layers, width 768, 12 heads, context 256):

| Model | Unique parameters |
| --- | ---: |
| Existing `100m`, untied, with the 32K vocabulary | 149,760,000 |
| `100m_tied`, same shape and vocabulary | 124,594,176 |
| Saving: `32,768 × 768` | **25,165,824** |

The historical `100m` name describes the character model's approximate scale;
its total grows when replacing its small character vocabulary with 32K subwords.
This PR does not resize layers to force a 100M total, or reinvest the saving in
extra layers. The separate scholarly/code plan's approximately 150M target is
unchanged. Increasing context also changes learned-position parameter counts.

For a forward pass using the frozen tokenizer, from `training/`:

```python
import torch
from model_100m_tied.model import GPT, ModelConfig
from training_data.subword import load_frozen

tokenizer = load_frozen()
model = GPT(ModelConfig(vocab_size=tokenizer.get_vocab_size()))
ids = tokenizer.encode("def square(x):\n    return x * x\n").ids
with torch.no_grad():
    logits = model(torch.tensor([ids]))
assert model.lm_head.weight is model.token_embedding.weight
```

This example allocates the full model. Tests use tiny configurations for forward
and gradient checks and the meta device for full-size parameter counts.

## Integration boundary and checkpoint behavior

The model registry makes the variant available to existing architecture-aware
checkpoint loading and recipe selection. However, the general recipe trainer
and text-generation CLI still use **character tokenization**. The merged subword
preparation pipeline is separate; connecting its batches/checkpoint tokenizer
metadata to a model-training launcher remains future work. Selecting
`model="100m_tied"` in a Shakespeare recipe does not turn that dataset into BPE.
This PR verifies real frozen-tokenizer IDs through the model, not a completed
subword pretraining or text-generation workflow.

PyTorch checkpoint roundtrip and the existing `generate.load_model` helper
restore the shared Parameter and identical logits. Resume now checks model
identity, because tied/untied variants otherwise have identical config fields
and state-dict key shapes but different optimizer layouts and semantics. Resume
between them is rejected; legacy `100m`, `model_100m` and missing-name 100M
checkpoints remain equivalent. Intentional conversion of an old checkpoint is
not implemented. Start this comparison from random initialization.

Provenance records the tied wrapper plus inherited 100M configuration and 10M
block sources. The trainer's default behavior for architectures without declared
dependencies is unchanged.

## Validation and comparison

From `training/`:

```sh
uv sync --locked --extra corpus
uv run --extra corpus python -m unittest discover -s tests -p test_tied_embeddings.py -v
uv run --extra corpus ruff check .
uv run --extra corpus ruff format --check .
uv lock --check
```

Checks cover shared object identity, optimizer deduplication, preservation of
all other initial weights, dtype conversion, the sum of input/output gradient
contributions versus an untied reference, causality, checkpoint roundtrip,
cross-architecture resume rejection, legacy name compatibility, source hashes,
actual frozen-tokenizer IDs (including the highest valid index), and full-size
parameter counts. These are CPU correctness checks, not a training entrypoint.

For a quality comparison, train fresh tied and untied models with the same
frozen tokenizer/hash, prepared data/splits, shape, optimizer, schedule, token
budget, runtime and evaluation windows. Do not spend the saved parameters in
the same experiment. Compare validation loss versus token exposure and elapsed
time, throughput and peak memory; repeat promising results across paired seeds.
No new training run, quality comparison, or GPU allocation is part of this PR.
