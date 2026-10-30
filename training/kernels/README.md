# Fused training and rollout kernels

Custom CUDA kernels for the 124M decoder, measured on one H100. Pretraining does not use
the training kernels yet; RL rollouts use the decode kernels ([below](#rollout-decoding)). Run from `training/` with Modal credentials:

```sh
uv run --all-extras modal run -m kernels.modal_profile [--fused-kernels]  # where step time goes
uv run --all-extras modal run -m kernels.modal_bench                      # correctness + timing
```

Each launch is one function call on one H100 (30-minute timeout, no retries) that refuses
to start once 24 hours have passed since launch. [`image.py`](image.py) is a CUDA 13.0
devel image with torch 2.14.1 so the extension compiles with `nvcc`; outputs go to the
ignored `artifacts/`.

## What `fuse(model)` does

[`fused.py`](fused.py) replaces a GPT's forward without changing parameters or module
names, so checkpoints are interchangeable with the unfused model. It expects BF16 autocast.

| Kernel ([`fused_ops.cu`](fused_ops.cu)) | Forward | Backward |
| --- | --- | --- |
| RMSNorm | FP32 residual in, BF16 out, one warp per row | input and weight gradients |
| Residual add + RMSNorm | residual + sublayer output + its bias, then the next norm | residual, sublayer, bias and weight gradients |
| QKV split + RoPE | adds the QKV bias, rotates by each token's position within its document, writes q, k and v in attention layout | input and bias gradients |
| SwiGLU | adds both biases, `silu(gate) * up`, eight BF16 values per thread | input and both bias gradients |
| LM head + cross-entropy | logits, loss and gradient 4,096 rows at a time, gradient written over the logits | precomputed in the forward |

Linears run without bias; the kernel that consumes each output adds the bias and sums its
gradient in registers, so no separate bias reductions run. Attention is the model's own
document-causal call (FlashAttention's varlen kernel), so packed documents stay isolated
exactly as in the unfused model.

## Results with document masking (one H100, torch 2.14.1, 2026-10-25)

16 x 2,048-token windows with an EOS every 1,000 tokens, BF16 autocast, fused AdamW.
Every variant masks attention by document.

| Step variant | Time | Tokens/s | Peak memory |
| --- | ---: | ---: | ---: |
| Eager PyTorch | 151 ms | 217K | 33.5 GB |
| `torch.compile` | 88.4 ms | 371K | 18.2 GB |
| Fused kernels | 83.7 ms | 392K | 16.0 GB |

Every kernel, including QKV + RoPE with positions restarting at document boundaries,
matches PyTorch to BF16 rounding; whole-model gradients agree with the unfused model at
cosine 0.99995. Attention alone (forward + backward, 16 x 2,048 x 12 heads x 64):
causal SDPA 1.57 ms, varlen with ~1,000-token documents 1.59 ms, varlen with one
document per row 2.36 ms. Reports: HF dataset `jfrick/lm-exploration-results`,
`training/results/document-masking-2026-10-25/`.

## Results without document masking (one H100, 16 x 2,048 tokens, BF16 autocast, fused AdamW, torch 2.6.0, 2026-10-24)

| Step variant | Time | Tokens/s | Peak memory |
| --- | ---: | ---: | ---: |
| Eager PyTorch | 165 ms | 198K | 33.0 GB |
| `torch.compile` | 96 ms | 342K | 19.0 GB |
| Fused kernels, FlashAttention-2 | 88.6 ms | 370K | 16.7 GB |
| Fused kernels, cuDNN attention | 81.9 ms | 400K | 17.2 GB |

Every kernel matches PyTorch to BF16 rounding, including bias gradients; whole-model
gradients agree with the unfused model at cosine 0.99995.

In the eager step about 60% of GPU time was element-wise work. With the fused kernels,
the linear layers run at about 700 TFLOPS (roughly 70% of BF16 peak), and matmuls plus
attention are most of the step.

### FP8 linears: slower and less accurate at this width

Block linears on FP8 tensor cores through `torch._scaled_mm`, with one dynamic scale per
tensor (e4m3 activations and weights, e5m2 gradients) and the LM head in BF16, took
169 ms per step with FlashAttention-2 and 163 ms with cuDNN. Gradients drifted to cosine
0.980 against BF16 (single linear: 4% forward, 6% gradient relative error). At width 768
the scaling passes and the transposed copies the backward needs cost more than the faster
matmuls save. FP8 would need fused quantize-and-transpose kernels, finer (per-row)
scales and a training run showing unchanged loss.

## Rollout decoding

RL samples hundreds of answers per step one token at a time, which in eager PyTorch is
bound by kernel launches and masked attention. [`decode.py`](decode.py) captures one
whole decode step (all 14 blocks, the LM head and sampling) as a CUDA graph and replays
it per token (`trainer.rollout.sample(..., engine="graph")`, the RL default). The graph
uses the RMSNorm kernel above, cuBLAS linears on BF16 weight copies, and three kernels
from [`decode_ops.cu`](decode_ops.cu) that read the cache slot, step and positions from
device memory, so shapes stay fixed across steps:

| Kernel | Work |
| --- | --- |
| `rope_append` | RoPE of the new query and key with the model's BF16 rounding; writes key and value to the cache slot |
| `attention` | one query per (row, head) over the row's cached keys, online softmax, eight warps per block |
| `sample` | log-sum-exp of logits / T, inverse-CDF draw from a pre-drawn uniform, the token and its log-probability |

Prefill runs eagerly into the same cache. Graphs are cached per (batch, capacity, top-k,
greedy) with capacity rounded up to 256 tokens, and weights are copied into the graph's
buffers before every rollout.

```sh
uv run --all-extras modal run -m kernels.modal_rollout::main --checkpoint RUN/step-N.pt
uv run --all-extras modal run -m kernels.modal_rollout::check --checkpoint RUN/step-N.pt
```

### Results (one H100, r02 math SFT checkpoint, 256 rows, 100–200 prompt tokens, 2026-10-25)

| Engine | ms per decode step | With top-32 recorded |
| --- | ---: | ---: |
| Eager PyTorch | 10.6 | 10.6 |
| CUDA graph + decode kernels | 3.29 | 3.45 |

- **Correctness:** a graph-replayed step equals the same kernels called step by step
  (log-probability differences at most 1e-6), and the `sample` kernel matches
  `log_softmax` to 1e-6 and a reference inverse-CDF draw on every row.
- **Against the learner** (the full forward pass the RL loss uses), the sampled tokens'
  mean absolute log-probability difference is 0.0153 for the graph engine and 0.0154 for
  eager decoding. Greedy decoding gives identical 64-token answers on 89% of rows; the
  rest diverge at BF16 near-ties.
- **Where time goes:** the attention kernel is the largest share of a graph step,
  followed by the linears.
