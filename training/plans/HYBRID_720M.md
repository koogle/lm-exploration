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

A side pilot of speedrun-style n-gram hints on this architecture is not part of
the plan; see its [report](../references/n-gram-hints-2026-10-29.md).

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
| 8 x H100, fused loss + CUDA gradient kernel | synthetic, 1,000-token documents | 372,758 | 27.9% |
| 8 x H100, fused loss + CUDA gradient kernel | prepared recipe (40B run) | ~372,700 | 27.9% |

Microbatch 8 runs out of memory on an 80 GB H100. Eight GPUs give 7.9 times the
one-GPU throughput at the same 1,048,576-token global batch.

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
| Fourth launch | 2026-10-25 ~09:55 UTC, 8 x H100, resumed from step 1,466 (1.54B tokens) after the 124M run released its GPUs; uses the shared RMSNorm and SwiGLU |
| Interruption | 2026-10-26 04:28 UTC at step ~24,018: the first launch's deadline supervisor cancelled every call recorded under the run ID, including the fourth launch's segment (the launcher records calls per launch since #81's fix). Resumed 06:50 UTC from the step-23,405 checkpoint (~0.6B tokens repeated) |
| Fifth and sixth launches | 2026-10-26 06:50 UTC, 8 x H100, from step 23,405; stopped gracefully at step 26,142 (27.41B tokens) before the fourth launch's supervisor fired at 09:39; relaunched 09:53 UTC from step 26,142, all from commit `24692d1` (branch `claude/hybrid-720m-40b-01-run`) |
| Throughput | median 371.8K tokens/s on 8 x H100, 27.9% MFU |
| Finished | 2026-10-26 19:55 UTC at step 38,146 (39,998,980,096 tokens, `stop_reason: target`); final bf16 export `tokens-039998980096/` in `jfrick/hybrid-720m-40b` |
| Checkpoints | every 90 minutes and at each stop; bf16 exports to `jfrick/hybrid-720m-40b` (private) every 5B tokens |
| Data | `capacity.json` and `screen.json` in `training/results/hybrid-40b-v1/` of the results dataset |

The workspace allows 10 concurrent GPUs, and the 124M run held 8 until its own
deadline, so the first segments ran on the 2 free H100s. Moving between GPU
counts uses:

```sh
uv run modal run -m workflows.modal_train::stop --run-id <id>
uv run modal run -m model_hybrid.modal_tools::bench --gpus 8
uv run modal run --detach -m workflows.modal_train::main --workflow workflows/hybrid-h100x8.json \
    --run-id <id> --start pretrain
```

Launching the workflow run again continues its unfinished stage. The resumed segment keeps the 1,048,576-token global batch. Every launch stops
within 24 hours.

Held-out loss (bits per byte); the learning rate is constant until 34B tokens and
then decays linearly to 0:

| Step | Tokens | ClimbMix | Code | Math |
| ---: | ---: | --- | --- | --- |
| 2,500 | 2.6B | 3.051 (0.912) | 1.510 (0.546) | 2.224 (0.886) |
| 5,000 | 5.2B | 2.961 (0.885) | 1.409 (0.510) | 2.142 (0.853) |
| 10,000 | 10.5B | 2.913 (0.871) | 1.400 (0.506) | 2.106 (0.839) |
| 15,000 | 15.7B | 2.891 (0.864) | 1.345 (0.487) | 2.082 (0.830) |
| 26,142 | 27.4B | 2.872 (0.859) | 1.357 (0.491) | 2.061 (0.821) |
| 32,000 | 33.6B | 2.862 (0.856) | 1.332 (0.482) | 2.059 (0.820) |
| 36,000 | 37.7B | 2.713 (0.811) | 1.191 (0.431) | 1.898 (0.756) |
| 38,146 (final) | 40.0B | **2.558 (0.765)** | **1.055 (0.382)** | **1.742 (0.694)** |

The decay phase (the last 6B tokens) lowered held-out bits per byte by 11% on
ClimbMix, 21% on code and 15% on math. Final MTP held-out loss: 2.80 / 1.23 / 1.96.
Greedy samples at 27.4B tokens were fluent but factually loose and repetitive
(for example "The first printing press was invented in 1796 by Thomas Jefferson").
At 40B the same prompts give a Gutenberg-centred printing-press paragraph, the
correct recursive `fibonacci(n-1)+fibonacci(n-2)` (repeated), the train question
worked as 120 km / 2 h = 60 km/h, and a prime-number definition with 2 and 3 as the
first primes. Metrics, samples, configs, launch records and segment logs:
`training/results/hybrid-720m-40b-01/` in `jfrick/lm-exploration-results`.

Compute, from the launch records and segment end times: 256.1 H100-hours over six
segments (17.1 on 2 GPUs, 239.0 on 8), 35.4 segment-hours of wall time. At Modal's
list $3.95 per H100-hour that is about $1,012 of GPU time plus about $127 for each
segment's 32 CPU cores and 256 GiB of memory, about $1,140 in total; the
interruption repeated about 0.6B tokens (about 3.5 H100-hours).

Cost estimate (Modal list price $3.95 per H100-hour, 2026-10-25): at the
measured ~46K tokens/s per GPU, 40B tokens take about 240 H100-hours, about $950,
plus roughly $3.60 per wall-clock hour for the segment's 32 CPU cores and 256 GiB
of memory (about $110 over 30 hours on 8 GPUs). Data preparation used 718 calls of
8 cores for about 17 minutes of wall time. The final measured MFU, GPU-hours and
cost are recorded above.

Studies that spend compute or touch data at scale (token counts, contamination
screens, benchmarks, pilots, the launch) are run only after the user approves
each one.

## Benchmarks

lm-evaluation-harness 0.4.13 through [`modal_benchmarks.py`](../hybrid/modal_benchmarks.py)
(2026-10-27): the same tasks, shot counts and greedy decoding for every model;
reference models pinned by revision. Zero-shot except MMLU (5-shot), GSM8K (5-shot,
flexible extract) and MBPP (3-shot); acc_norm where the harness reports it. Raw
results: `training/results/hybrid-720m-40b-01/benchmarks/` in
`jfrick/lm-exploration-results`.

| Model | Tokens | HellaSwag | ARC-E | ARC-C | PIQA | WinoG | OBQA | SciQ | LAMBADA | MMLU | GSM8K | HumanEval | MBPP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid 720M (40B export) | 40B | 51.4 | 63.4 | 35.4 | 72.7 | 57.4 | 38.8 | 87.8 | 41.2 | 26.4 | 3.5 | 4.3 | 5.0 |
| Gemma 3 270M | 6T | 41.3 | 57.1 | 28.3 | 68.1 | 53.4 | 30.6 | 83.8 | 43.0 | 26.3 | 2.2 | 3.0 | 1.8 |
| SmolLM2-360M | 4T | 56.3 | 68.2 | 38.5 | 71.5 | 59.4 | 38.0 | 86.5 | 53.3 | 25.3 | 3.9 | 0.0 | 19.8 |
| Qwen3-0.6B-Base | ~36T | 53.8 | 58.1 | 38.1 | 70.1 | 58.6 | 34.4 | 91.5 | 53.8 | 52.5 | 50.9 | 29.9 | 38.8 |
| Qwen3.5-0.8B-Base | | 54.8 | 67.5 | 40.4 | 71.5 | 60.7 | 36.0 | 91.8 | 50.8 | 48.3 | 40.0 | 22.6 | 25.6 |
| Gemma 3 1B | 2T | 62.1 | 72.0 | 38.2 | 74.7 | 58.5 | 37.6 | 92.1 | 55.9 | 26.3 | 2.7 | 6.1 | 8.8 |
| OLMo-2-0425-1B | 4T | 68.3 | 73.6 | 42.2 | 75.8 | 64.4 | 39.4 | 95.1 | 64.9 | 42.4 | 33.7 | 6.7 | 7.6 |
| LFM2.5-1.2B-Base | | 59.2 | 77.7 | 51.3 | 72.9 | 59.5 | 39.0 | 93.2 | 46.6 | 55.1 | 55.0 | 23.8 | 23.2 |
| Gemma 4 E2B | | 73.7 | 79.9 | 52.0 | 78.7 | 64.7 | 44.6 | 95.9 | 67.1 | 57.7 | 23.4 | 26.8 | 36.8 |

The run's own exports:

| Tokens | HellaSwag | ARC-E | ARC-C | PIQA | WinoG | OBQA | SciQ | LAMBADA | MMLU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5B | 37.5 | 54.3 | 27.5 | 68.0 | 50.4 | 31.8 | 75.1 | 23.5 | 25.3 |
| 10B | 39.1 | 54.5 | 30.5 | 67.7 | 51.5 | 33.4 | 81.7 | 31.0 | 24.4 |
| 15B | 39.6 | 54.5 | 28.6 | 68.4 | 51.0 | 32.2 | 79.1 | 26.7 | 25.0 |
| 20B | 40.5 | 56.3 | 30.4 | 69.6 | 52.6 | 34.2 | 81.4 | 26.4 | 24.3 |
| 25B | 40.7 | 55.4 | 28.5 | 69.0 | 54.5 | 33.2 | 82.1 | 33.7 | 26.1 |
| 30B | 40.3 | 54.5 | 28.7 | 68.3 | 52.8 | 33.0 | 80.7 | 33.9 | 23.6 |
| 35B | 41.8 | 56.3 | 32.0 | 70.2 | 53.0 | 34.0 | 81.9 | 31.5 | 28.1 |
| 40B | 51.4 | 63.4 | 35.4 | 72.7 | 57.4 | 38.8 | 87.8 | 41.2 | 26.4 |

At constant learning rate (to 34B tokens) the benchmarks rose slowly; the
decay over the last 6B tokens added 10 points on HellaSwag and LAMBADA and 7 on
ARC-E. The model is ahead of Gemma 3 270M, close to SmolLM2-360M and Qwen3-0.6B on
commonsense tasks (ahead on PIQA and OpenBookQA), and 5–10 points behind the
1B-class models there. MMLU, GSM8K and code are at chance or near zero, as for
Gemma 3 1B and SmolLM2; the models at ~50% on MMLU and GSM8K (Qwen3, Qwen3.5,
LFM2.5) trained on far more, and more exam-like, data. The training data was
screened against these test sets; the reference models' was not necessarily.

