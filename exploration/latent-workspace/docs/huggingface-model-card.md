---
language:
- en
- multilingual
base_model: Qwen/Qwen2.5-0.5B
tags:
- speculative-decoding
- diffusion
- research
- pytorch
---

# Frozen-Qwen 79M diffusion-style drafter

A separately trained 79M drafting head for frozen **Qwen2.5-0.5B**. This is a
custom research architecture, requiring Qwen's tokenizer, embeddings, output
weights and context hidden states. It cannot generate as a standalone causal LM
or load with `AutoModelForCausalLM`.

Selected checkpoint: step **70,585**, run `fresh-grid-196608-r256-11`.
Original checkpoint SHA256:
`44865d84016898f575cf009c5cd2c2acb919564f009e912170cf4ba131b4e768`.
The inference-only `drafter.pt` has a different checksum, recorded in
`provenance.json`. It contains draft weights, configuration, mask token ID and
step, without optimizer state, datasets, or frozen target weights.

## Architecture and training

Five drafting layers with query/key normalization condition on Qwen post-block
hidden states 2, 7, 12, 17 and 22. Each block contains one target-supplied anchor
and seven parallel masked proposals. A single pass predicts the seven new tokens;
this is not iterative denoising. The target stays frozen throughout training.

Training used Qwen-generated responses capped at 256 tokens, beginning with
131,072 prose/code examples and continuing to 196,608 balanced prose/code/
translation examples. AdamW, learning rate 0.0006, weight decay 0.01, 4% warmup,
cosine decay, gradient clipping at 1, seed 42. Cross-entropy weights earlier
positions more heavily (gamma 4); the anchor is excluded from the loss.

## Measured results and limitations

Validation-selected mean consecutive correct NEW tokens: **4.033 / 7**.
One sealed test: **3.938 / 7** on 189 usable prompts (63/domain), with prose
**5.014**, code **4.958**, translation **1.843**. This is one trained seed; the
original test is consumed and must not select subsequent variants.

All 24 longer verified generations (3,906 total tokens) exactly matched greedy
Qwen output using FP32 verification on stored BF16 target weights. BF16
AR/block parity is not established. These results establish drafting
prediction quality under the study protocol, not better answer quality or a
wall-clock speedup. Qwen prose/code responses can be repetitive.

## Usage

Download a pinned revision of this model repository and install `requirements.txt`.
The bundled code avoids depending on access to the private research repository.
Use the commit revision returned by the publishing command instead of `main`.

```python
from huggingface_hub import snapshot_download

folder = snapshot_download("jfrick/qwen2.5-0.5b-dflash-drafter-79m", revision="COMMIT_SHA")
```

From the downloaded folder:

```bash
python -m pip install -r requirements.txt
python -c 'from dflash_cli import generate_checkpoint; print(generate_checkpoint("drafter.pt", "The capital of France is", 32)["text"])'
```

The code downloads target revision
`060db6499f32faf8b98477b0a26969ef7d8b9987` and verifies proposed blocks against
that frozen model. CUDA is used when available; CPU inference is supported but
slow. Review downloaded Python before running it.

## Sources and licensing

[LM Exploration](https://github.com/koogle/0-to-llm),
[DFlash](https://arxiv.org/abs/2602.06036),
[reference implementation](https://github.com/z-lab/dflash).
Draft architecture adapted from MIT-licensed DFlash commit
`07ebd93db9f472af339b644bb70221ad8428328a`; its notice is in `dflash-LICENSE`.
Qwen weights are downloaded separately under their upstream license. No broader
license grant for this project's weights or original code has been specified.
