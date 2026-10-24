# Hybrid model family

`hybrid/` (installed as `model_hybrid`) holds the decoder for the
[~720M hybrid pretraining plan](../plans/HYBRID_720M.md). It is separate from
`10m/` and `100m/`, so the 124M Simple LLM runs use unchanged code.

## Backbone

[`model.py`](model.py) defines `HybridConfig` and `HybridLM`:

- Pre-norm blocks with `RMSNorm` (statistics in fp32) and a `SwiGLU`
  feed-forward layer, both the shared classes from [`10m/model.py`](../10m/model.py);
  the hybrid family builds `SwiGLU` without biases at a configurable hidden width.
- Tied input/output embeddings; no bias parameters anywhere.
- Full-attention layers use grouped-query attention (`n_head` query heads,
  `n_kv_head` key/value heads, head dimension 128), RMSNorm on each query and key
  head (QK-norm), and RoPE with a configurable base (default 1,000,000).
- Initialization: Normal(0, 0.02) for embeddings and projections; residual
  output projections (attention `o_proj`, SwiGLU `down`) use
  `0.02 / sqrt(2 * n_layer)`. Norm scales start at 1.
- `forward(tokens)` returns logits. `forward(tokens, labels)` returns summed
  cross-entropy `ce`, summed squared log-normalizer `z`, `loss = ce + 1e-4 * z` and
  the valid-target `count`. Logits are computed in chunks of `loss_chunk_size`
  rows and recomputed in backward, so the full `[tokens, vocab]` logit tensor is
  never stored. Labels of `-100` are ignored.

## Document masking

Loaders pack documents back to back, each ending in EOS.
[`documents.py`](documents.py) derives per-row document ids from those EOS
boundaries (`document_ids`), converts them to flattened `cu_seqlens` for varlen
kernels, and `DocumentBatches` wraps any token mixture so `next_batch` returns
`(inputs, labels, document_ids)`. Pass the ids as `forward(tokens, labels, documents)`.

Attention is document-causal: a token sees earlier tokens of its own document
only, and RoPE positions restart at each document. Without document ids each row
is treated as one document.

## Attention kernel

On CUDA, full attention runs our own kernel,
[`kernels/document_attention.cu`](kernels/document_attention.cu), wrapped as
`torch.library` custom ops in [`document_attention.py`](document_attention.py)
so `torch.compile` treats forward and backward as single nodes. It takes bf16 or
fp32 q/k/v (GQA, head dimension 64 or 128) and int32 document starts (row `i`
attends to keys `start_i..i`) and returns the output and the fp32 log-sum-exp.

The kernel is **bit-exact** with [`exact_attention.py`](exact_attention.py), a
PyTorch definition of every float operation and its order: dot products sum over
the head dimension in order, softmax sums and value accumulations sum over keys in
order, gradient accumulations over query heads and then query rows in order. All
operations are correctly rounded float32 multiplies, adds and divides (no fused
multiply-add; built with `-fmad=false`), and `exp_fixed`/`log_fixed` are short
polynomials, so the definition can be ported to other hardware (for example Metal)
with the same bits. `exp_fixed` has relative error below `1.2e-7 * max(1, |x|)`.
Each block owns disjoint outputs, so there are no atomics and repeated runs give
identical bits.

The forward makes two passes over the allowed key tiles of each 64-row query tile
(row maxima, then exponentials, sums and the value accumulation); the backward
recomputes probabilities from the saved maxima and sums, with one kernel for dQ
and one per key/value head for dK and dV. Tiles outside a row's document are
skipped. The arithmetic uses CUDA cores in float32, not tensor cores, whose
internal accumulation order cannot be reproduced exactly.

Measured on one H100 (torch 2.14.1+cu130, bf16, B=8, 12 query / 4 KV heads,
T=4096, head dimension 128): forward 49.8 ms and forward+backward 161 ms with one
document per row; 13.3 ms and 44.1 ms with four documents per row. PyTorch's
causal SDPA (tensor cores, one document) takes 0.69 ms and 3.1 ms. Exactness
costs this factor; a faster kernel must keep the same reference bits.

On CPU, `model.reference_attention` (an explicit masked softmax, fp32 or fp64) is
the model's attention.

The defaults (width 1536, 16 layers, 12 query / 4 KV heads, SwiGLU 4096) are
placeholders until the 720M configuration is fixed in PR 8.

## Tests

From `training/`:

```sh
uv run python -m unittest tests.test_hybrid_backbone tests.test_hybrid_documents \
  tests.test_hybrid_exact_attention -v
```

On one H100 (30-minute timeout, no retries, refuses work 24 hours after launch):

```sh
uv run modal run -m model_hybrid.modal_checks
```

The Modal image uses the CUDA 13.0 devel base and prebuilds the kernel.
[`check_hybrid_cuda.py`](../tests/check_hybrid_cuda.py) requires the kernel's
outputs, statistics and gradients to equal `exact_attention` bit for bit (fp32 and
bf16, GQA, packed documents, lengths that are not tile multiples, head dimensions
64 and 128) and to repeat bit for bit, reports the distance to fp64 softmax
attention, runs a compiled bf16 model with `fullgraph=True`, and times the kernel.

`test_hybrid_exact_attention` checks `exp_fixed` and `log_fixed` accuracy, and
the reference's output, log-sum-exp and gradients against fp64 autograd.

`test_hybrid_documents` checks document ids, positions and `cu_seqlens`, and that
document B's logits are identical whether it follows a different document A, or
none. The backbone tests cover output shapes, tied weights and absent biases, initialization scales,
fp32 RMSNorm statistics, causality (changing future tokens leaves earlier logits
bit-identical), attention against an explicit softmax, GQA against MHA with
duplicated key/value heads, the RoPE base and the chunked loss against
`F.cross_entropy` plus an explicit z-loss.
