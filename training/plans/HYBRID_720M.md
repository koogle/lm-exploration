# ~720M hybrid pretraining plan

A ~0.72B-parameter decoder mixing Kimi Delta Attention (KDA) and full-attention
layers, pretrained on 40B tokens with the GLM tokenizer. The model lives in its
own family, [`hybrid/`](../hybrid/README.md), so the running 124M Simple LLM jobs
are unaffected.

The work lands as a stack of pull requests, each based on the previous one:
architecture first, data and launch last. No GPU training runs before PR 8 other
than small CUDA correctness checks.

| PR | Scope | State |
| --- | --- | --- |
| 1 | Backbone: RMSNorm, SwiGLU, tied embeddings, GQA, QK-norm, RoPE base 1M, scaled init, chunked CE + z-loss | Implemented, CPU-tested |
| 2 | Document masking: loader document ids / `cu_seqlens`; our own bit-exact document-causal attention kernel | Implemented, CPU-tested, H100 bit-exact check passed |
| 3 | Attention sink and sigmoid output gate | Implemented, CPU-tested |
| 4 | KDA layers via flash-linear-attention, document state reset, `[KDA, KDA, KDA, full]` layout | Implemented, CPU-tested, H100 check passed |
| 5 | One DeepSeek/GLM-style MTP module predicting t+2 | Implemented, CPU-tested |
| 6 | Muon/AdamW parameter grouping | Implemented, CPU-tested |
| 7 | GLM-4.7-Flash tokenizer and 40B recipe (70% ClimbMix, 20% code, 10% math) | Implemented; preparation running on Modal |
| 8 | 720M config, throughput benchmark, ~200M-token pilot | Implemented; 1- and 2-GPU benchmarks and the pilot measured |
| 9 | 40B launch on 8×H100 in 24-hour segments | Planned |

## Measurements (2026-10-25)

Throughput of the real training step (bf16, `torch.compile`, FlexAttention,
fused output-layer loss, microbatch 4 x 4,096), MFU against 989 TFLOPS per H100:

| Setup | Data | Tokens/s | MFU |
| --- | --- | ---: | ---: |
| 1 x H100, chunked autograd loss | synthetic, 1,000-token documents | 39,416 | 23.6% |
| 1 x H100, fused loss | synthetic, 1,000-token documents | 45,374 | 27.2% |
| 2 x H100, fused loss | prepared recipe (pilot) | ~92,600 | 27.8% |

Microbatch 8 runs out of memory on an 80 GB H100. The 8 x H100 benchmark could
not be scheduled: the workspace allows 10 concurrent GPUs and the 124M run holds
8 until its launch deadline.

Studies that spend compute or touch data at scale (token counts, contamination
screens, benchmarks, pilots, the launch) are run only after the user approves
each one.
