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
| 2 | Document masking: document ids from EOS / `cu_seqlens`; FlexAttention with a document-causal block mask | Implemented |
| 3 | Attention sink and sigmoid output gate | Implemented, CPU-tested |
| 4 | KDA layers via flash-linear-attention, document state reset, `[KDA, KDA, KDA, full]` layout | Implemented, CPU-tested, H100 check passed |
| 5 | One DeepSeek/GLM-style MTP module predicting t+2 | Implemented, CPU-tested |
| 6 | Muon/AdamW parameter grouping | Implemented, CPU-tested |
| 7 | GLM-4.7-Flash tokenizer and 40B recipe (70% ClimbMix, 20% code, 10% math) | Implemented; preparation running on Modal |
| 8 | 720M config, throughput benchmark, ~200M-token pilot | Implemented; 1- and 2-GPU benchmarks and the pilot measured |
| 9 | 40B launch on 8×H100 in 24-hour segments | Running (`hybrid-720m-40b-01`) |

## Measurements (2026-10-25)

Throughput of the real training step (bf16, `torch.compile`, FlexAttention,
fused output-layer loss, microbatch 4 x 4,096), MFU against 989 TFLOPS per H100:

| Setup | Data | Tokens/s | MFU |
| --- | --- | ---: | ---: |
| 1 x H100, chunked autograd loss | synthetic, 1,000-token documents | 39,416 | 23.6% |
| 1 x H100, fused loss | synthetic, 1,000-token documents | 45,374 | 27.2% |
| 2 x H100, fused loss | prepared recipe (pilot) | ~92,600 | 27.8% |
| 1 x H100, fused loss + Triton gradient kernel | synthetic, 1,000-token documents | 46,680 | 28.0% |
| 2 x H100, fused loss + Triton gradient kernel | prepared recipe (40B run) | ~96,000 | 28.7–29.0% |
| 1 x H100, fused loss + CUDA gradient kernel | synthetic, 1,000-token documents | 47,379 | 28.4% |
| 2 x H100, fused loss + CUDA gradient kernel | prepared recipe (40B run) | 100,000–104,000 | 29.9–31.3% |

Microbatch 8 runs out of memory on an 80 GB H100. The 8 x H100 benchmark could
not be scheduled: the workspace allows 10 concurrent GPUs and the 124M run holds
8 until its launch deadline.

## Pilot

`hybrid-720m-pilot-01` (2 x H100, the 40B config with frequent evaluation)
trained 104.9M tokens, checkpointed, and a fresh launch resumed it to 209.7M
tokens; the loss is continuous across the resume, with no non-finite values.
Held-out at 209.7M tokens: ClimbMix 5.42 nats/token (1.62 bits/byte), code 5.53
(2.00), math 4.85 (1.93). Gates are about 94% open; sinks take 0.4–0.9% of the
attention mass. Curves, metrics and config: HF dataset
`jfrick/lm-exploration-results`, `training/results/hybrid-720m-pilot-01/`.

## 40B run record

| | |
| --- | --- |
| Run ID | `hybrid-720m-40b-01` on volume `lm-exploration-hybrid-runs` |
| Config | [`configs/hybrid-720m-40b.json`](../hybrid/configs/hybrid-720m-40b.json), recipe `hybrid-40b-v1` |
| First launch | 2026-10-25 04:44 UTC, 2 x H100, commit recorded in `launches/`; stopped at step 46 to switch to the Triton gradient kernel |
| Second launch | 2026-10-25 ~05:15 UTC, 2 x H100, resumed from step 46; losses continue the pilot's trajectory |
| Third launch | 2026-10-25 ~05:45 UTC, 2 x H100, resumed from step 95 with the CUDA gradient kernel |
| Throughput | 100–104K tokens/s on 2 x H100 (30–31% MFU) |
| Checkpoints | every 90 minutes and at each stop; bf16 exports to `jfrick/hybrid-720m-40b` (private) every 5B tokens |
| Data | `capacity.json` and `screen.json` in `training/results/hybrid-40b-v1/` of the results dataset |

The workspace allows 10 concurrent GPUs, and the 124M run held 8 until its own
deadline, so the first segment runs on the 2 free H100s. When 8 are free:

```sh
uv run modal run -m model_hybrid.modal_train::stop --run-id hybrid-720m-40b-01
uv run modal run -m model_hybrid.modal_train::bench --gpus 8
uv run modal run --detach -m model_hybrid.modal_train::launch --run-id hybrid-720m-40b-01 --gpus 8
```

The resumed segment keeps the 1,048,576-token global batch. Every launch stops
within 24 hours; continuing needs a new launch with the same run ID.

Cost estimate (Modal list price $3.95 per H100-hour, 2026-10-25): at the
measured ~46K tokens/s per GPU, 40B tokens take about 240 H100-hours, about $950,
plus roughly $3.60 per wall-clock hour for the segment's 32 CPU cores and 256 GiB
of memory (about $110 over 30 hours on 8 GPUs). Data preparation used 718 calls of
8 cores for about 17 minutes of wall time. The final measured MFU, GPU-hours and
cost are recorded here when the run completes.

Studies that spend compute or touch data at scale (token counts, contamination
screens, benchmarks, pilots, the launch) are run only after the user approves
each one.
