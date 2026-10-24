# Hybrid model family

`hybrid/` (installed as `model_hybrid`) holds the decoder for the
[~720M hybrid pretraining plan](../plans/HYBRID_720M.md). It is separate from
`10m/` and `100m/`, so the 124M Simple LLM runs use unchanged code.

## Backbone

[`model.py`](model.py) defines `HybridConfig` and `HybridLM`:

- Pre-norm blocks with `RMSNorm` (statistics in fp32) and a `SwiGLU`
  feed-forward layer, both the shared classes from [`10m/model.py`](../10m/model.py);
  the hybrid family builds `SwiGLU` without biases at a configurable hidden width.
- Tied input/output embeddings; no bias parameters except the attention output
  gate's (below).
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

## Attention sink and output gate

Each full-attention layer has a learned sink: one logit per head, initialized to
0, acting as an extra key with a zero value. Softmax over the scores plus the sink
equals the plain attention output times `sigmoid(lse - sink)`, where `lse` is the
log-sum-exp of the scores, so the layer applies that factor to the kernel's (or
the reference's) output; the kernel's backward takes the log-sum-exp gradient. A head can then attend to "nothing" without spending
probability on a real token.

After the sink, an elementwise sigmoid gate multiplies the merged heads before
`o_proj`: `sigmoid(gate_proj(x) + gate_bias)`, with `x` the layer's normalized
input. `gate_bias` (per channel) starts at `gate_bias_init = 3.0`, so the gate
starts about 95% open. With the gate fully open and the sink at -inf the layer
reduces exactly to plain document-causal attention.

## Kimi Delta Attention and the hybrid layout

[`kda.py`](kda.py) implements KDA layers following flash-linear-attention's
`KimiDeltaAttention` ([Kimi Linear](https://arxiv.org/abs/2510.26692)), without
biases:

- q/k/v projections to `n_kda_head` heads of width `head_dim`, each followed by a
  causal depthwise convolution (kernel `kda_conv_size = 4`) and SiLU.
- L2-normalized q and k (q further scaled by `1/sqrt(head_dim)`), sigmoid beta
  from `b_proj`, and a per-channel decay `-exp(A_log) * softplus(f_proj(x) + dt_bias)`
  with a low-rank `f_proj`.
- Per-head output RMSNorm, a sigmoid gate from a low-rank `g_proj`, and `o_proj`
  (a residual output projection for initialization).
- No positional encoding; the recurrent state is fp32.

Document boundaries reset both the convolutions (they never read a previous
document's tokens) and the recurrent state. On CUDA the recurrence is fla's
`chunk_kda`, run over the flattened batch with `cu_seqlens` from the document ids;
gate, beta and L2 normalization run inside the kernel. On CPU, `kda_reference`
is an explicit per-token recurrence with fp32 (or fp64) state.

`torch.compile` traces through the fla call. On one H100, a small compiled hybrid
model (4 layers, width 512, 4 x 1024 tokens, bf16) took 18.7–19.1 ms per training
step that way and 19.0–19.5 ms with fla excluded by `torch.compiler.disable`
(three alternating rounds after warming up both), so the graph stays unbroken.

`HybridConfig.layer_pattern` lists `"kda"` and `"full"` layers and repeats to
`n_layer`; the default `("kda", "kda", "kda", "full")` puts full attention at
layers 3, 7, 11 and 15 of 16. Full-attention layers keep RoPE, the sink and the
output gate.

The defaults (width 1536, 16 layers, 12 query / 4 KV heads, SwiGLU 4096) are
placeholders until the 720M configuration is fixed in PR 8.

## Tests

From `training/`:

```sh
uv run python -m unittest tests.test_hybrid_backbone tests.test_hybrid_documents \
  tests.test_hybrid_exact_attention tests.test_hybrid_sink_gate tests.test_hybrid_kda -v
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

The same check compares fla's `chunk_kda` with `cu_seqlens` against
`kda_reference` (output and all input gradients, within TF32 tolerance) and times
bf16 training steps of a small compiled hybrid model with the fla call outside the
compiled graph (`torch.compiler.disable`) and inside it, after warming up both.
The CUDA path needs the `hybrid` extra (`fla-core`); the Modal image installs it.

`test_hybrid_kda` compares the KDA recurrence with an independent matrix form
`S_t = (I - b k k^T) diag(a) S_(t-1) + b k v^T`, the short convolution with
`F.conv1d`, checks state and convolution resets at document boundaries, the
layout, document isolation and causality through mixed KDA/full stacks, and that
every parameter receives a gradient. `test_hybrid_sink_gate` compares the sink with an explicit softmax that has an
extra zero-value key, checks that an open gate and a -inf sink reproduce plain
attention exactly, and that the gate starts mostly open and receives gradients.
`test_hybrid_documents` checks document ids, positions and `cu_seqlens`, and that
document B's logits are identical whether it follows a different document A, or
none. The backbone tests cover output shapes, tied weights and absent biases, initialization scales,
fp32 RMSNorm statistics, causality (changing future tokens leaves earlier logits
bit-identical), attention against an explicit softmax, GQA against MHA with
duplicated key/value heads, the RoPE base and the chunked loss against
`F.cross_entropy` plus an explicit z-loss.
