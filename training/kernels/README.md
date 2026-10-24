# Fused training kernels

Custom CUDA kernels for the element-wise work of the 124M decoder, measured on one H100.
Training does not use them yet. Run from `training/` with Modal credentials:

```sh
uv run --all-extras modal run -m kernels.modal_profile [--fused-kernels]  # where step time goes
uv run --all-extras modal run -m kernels.modal_bench                      # correctness + timing
```

Each launch is one function call on one H100 (30-minute timeout, no retries) that refuses
to start once 24 hours have passed since launch. [`image.py`](image.py) is a CUDA 12.4
devel image so the extension compiles with `nvcc`; outputs go to the ignored `artifacts/`.

## What `fuse(model)` does

[`fused.py`](fused.py) replaces a GPT's forward without changing parameters or module
names, so checkpoints are interchangeable with the unfused model. It expects BF16 autocast.

| Kernel ([`fused_ops.cu`](fused_ops.cu)) | Forward | Backward |
| --- | --- | --- |
| RMSNorm | FP32 residual in, BF16 out, one warp per row | input and weight gradients |
| Residual add + RMSNorm | residual + sublayer output + its bias, then the next norm | residual, sublayer, bias and weight gradients |
| QKV split + RoPE | adds the QKV bias, writes rotated q, k and v in attention layout | input and bias gradients |
| SwiGLU | adds both biases, `silu(gate) * up`, eight BF16 values per thread | input and both bias gradients |
| LM head + cross-entropy | logits, loss and gradient 4,096 rows at a time, gradient written over the logits | precomputed in the forward |

Linears run without bias; the kernel that consumes each output adds the bias and sums its
gradient in registers, so no separate bias reductions run. Attention uses PyTorch's cuDNN
kernel by default (`fuse(model, attention="flash")` selects FlashAttention-2).

## Results (one H100, 16 x 2,048 tokens, BF16 autocast, fused AdamW, 2026-10-24)

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
