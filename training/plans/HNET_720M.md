# Byte-level H-Net vs the 720M GLM-tokenized hybrid

Does replacing the GLM tokenizer with learned dynamic chunking (H-Net) give a
better model than [`hybrid-720m-40b-01`](HYBRID_720M.md) for the same data and
training compute?

Motivation: Cartesia's
[From Tokens to Bytes](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1)
(2026-10-28) reports compute-efficiency gains for byte-level H-Nets over BPE
Transformers in controlled scaling ladders: 1.28x on in-domain validation loss and
about 2x on downstream averages. The improved architecture described there is not
released, so this study implements the published 2025 H-Net
([arXiv 2507.07955](https://arxiv.org/abs/2507.07955),
[goombalab/hnet](https://github.com/goombalab/hnet)). The original paper also
reports that byte-level H-Nets start behind the BPE model and overtake it with more
data.

## Design

[`hybrid/hnet.py`](../hybrid/hnet.py), config
[`hnet-720m-40b.json`](../hybrid/configs/hnet-720m-40b.json); details in the
[hybrid README](../hybrid/README.md#byte-level-h-net).

| | GLM hybrid (`hybrid-720m-40b-01`) | H-Net (`hnet-720m-cmp-02`) |
| --- | --- | --- |
| Input | GLM-4.7-Flash tokens (154,880 vocab) | UTF-8 bytes + EOS (257) |
| Main network | 16 layers, 4 x [KDA, KDA, KDA, full], width 1536 | the same, over chunks |
| Outer stage | embedding + tied output layer, MTP module | 4 + 4 KDA/SwiGLU blocks at width 1024, routing, dechunk |
| Parameters | 740.0M: embedding/output 237.9M, blocks 469.2M, MTP 32.8M (707.1M without MTP) | 580.1M: main network 469.2M, encoder + decoder 107.1M, routing, residual, byte embedding and output 3.7M |
| Training FLOPs per byte | 1.31G (5.94G per token / 4.54 bytes per token) | 1.29G at 4.5 bytes per chunk |
| Context | 4,096 tokens | 16,384 bytes (about 3,600 tokens) |
| Batch | 256 x 4,096 tokens (about 4.76 MB) | 288 x 16,384 bytes (4.72 MB) |
| Warmup | 500M tokens | 2.27 GB (the same data) |
| Optimizer | Muon/AdamW as in the hybrid README | the same |

The target compression ratio, 4.5 bytes per chunk, is GLM's measured bytes per
token on this mix (ClimbMix 4.83, code 3.99, math 3.62, token-weighted 4.54), so
the main network processes about as many positions per byte as it does with GLM
tokens. The H-Net trains on the same prepared shards, decoded back to bytes, in the
same order and document mix, and is evaluated on the same held-out documents;
bits per byte is directly comparable.

## Measured before launch (2026-10-29, H100, torch 2.14.1)

- CUDA check: compiled bf16 loss 5.5863 vs fp64 CPU reference 5.5858; document
  isolation within bf16 (max logit difference 0.008).
- Microbatch at 4 bytes per chunk (forced routing), one H100: 2 rows use 44 GiB at
  94K bytes/s, 3 rows 63 GiB at 100K bytes/s, 4 rows run out of memory. The run
  uses 2, leaving room for routing transients.
- 94K bytes/s per H100 is about 21K GLM-token equivalents per second, against
  46K tokens/s for the GLM model: the H-Net takes about 2.2x the wall-clock time for
  the same FLOPs. The byte-level KDA/SwiGLU layers run at about 10% MFU.
- Smoke run `hnet-720m-smoke-01` (1 H100, 4 steps, real data): loading, held-out
  evaluation, samples and checkpoints work; the initial routing gives about 50
  bytes per chunk before the ratio loss acts.

## Routing stability

The first launch, `hnet-720m-cmp-01` (Muon on the routing projections, ratio-loss
weight 0.03, microbatch 3), ran out of memory at step ~75 after about 20 minutes on
8 GPUs. Bytes per chunk went 50 (step 1), 1,748 (step 30), 2,566 (step 50), then 14
(step 70) and lower; with almost every byte a chunk, the main network no longer fit.

Four 1-GPU tests, each with 24 rows of 16,384 bytes, warmup over 481 steps
(as in the full run) and 300 steps planned:

| Routing optimizer | Ratio-loss weight | Bytes per chunk (step 30 / 60 / 90+) | Outcome |
| --- | ---: | --- | --- |
| Muon | 0.03 | 320 / 2,979 / 2.2 | out of memory at step ~95 |
| Muon | 0.3 | 259 / 186 (step 50) / 17.8 (step 55) | out of memory at step ~55 |
| AdamW | 0.03 | 328 / 2,092 / – | out of memory after step 60 |
| AdamW | 0.3 | 267 / 3.7 / 4.1–4.4 through step 210 | stable |

Every variant first loses almost all boundaries (the encoder's adjacent states
grow similar faster than the ratio loss responds), then swings back. Only AdamW
routing (as in the H-Net paper) with weight 0.3 settled near the 4.5 target. The
configs now use that, with microbatch 2 and expandable CUDA segments.

## Run

`hnet-720m-cmp-02` ran from commit `4fb7076` with that commit's launcher:
`modal_train::launch --run-id hnet-720m-cmp-02 --config hybrid/configs/hnet-720m-40b.json
--gpus 8 --max-tokens 23800000000`. H-Net runs now go through the workflows like every
family (`workflows/hnet-h100x8.json`, see the
[hybrid README](../hybrid/README.md#byte-level-h-net)); a workflow trains to the
config's `target_tokens`.

23.8 GB of bytes (5,043 steps) matches the GLM run's first 5,000 steps (5.24B
tokens). Both runs are then still at the constant learning rate, so held-out bits
per byte at matching steps compares the two at equal data and compute. The config's
`target_tokens` is the byte equivalent of 40B tokens, so the run's schedule is a
prefix of a full 40B-equivalent run.

| | |
| --- | --- |
| Run ID | `hnet-720m-cmp-02` on volume `lm-exploration-hybrid-runs` (final checkpoint there) |
| Launched | 2026-10-29 12:30 UTC, 8 x H100, commit `4fb7076`, one segment with a supervised 24 h deadline |
| Finished | about 21:15 UTC at step 5,043 (23,795,859,456 bytes, `stop_reason: token_limit`), no restarts |
| Throughput | 850–890K bytes/s on 8 GPUs, 14.8% MFU; 4.2–4.4 bytes per chunk in training after step 100 |
| Compute | about 70 H100-hours, about $280 at list price plus about $32 CPU and memory; with `cmp-01`, the routing tests, the smoke run, checks and benchmarks about $340 |
| Records | metrics, configs, launches and logs of every run here: `training/results/hnet-720m/` in the HF dataset `jfrick/lm-exploration-results` |
| Model | public bf16 weights, config, training log and model card: [`jfrick/hnet-720m`](https://huggingface.co/jfrick/hnet-720m) (`model_hybrid.modal_tools::publish`) |

## Results

Held-out bits per byte on the same documents, both at the constant learning rate
(H-Net steps are 1% larger in bytes than the GLM steps with the same number):

| Step | Bytes | GLM ClimbMix | H-Net ClimbMix | GLM code | H-Net code | GLM math | H-Net math |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 2.4 GB | 1.089 | 1.284 (+18%) | 0.802 | 0.955 (+19%) | 1.108 | 1.368 (+23%) |
| 1,000 | 4.7 GB | 0.974 | 1.078 (+11%) | 0.635 | 0.669 (+5%) | 0.970 | 1.080 (+11%) |
| 2,000 | 9.4 GB | 0.926 | 0.998 (+8%) | 0.578 | 0.614 (+6%) | 0.909 | 0.983 (+8%) |
| 3,000 | 14.2 GB | 0.902 | 0.951 (+5%) | 0.532 | 0.540 (+2%) | 0.876 | 0.919 (+5%) |
| 4,000 | 18.9 GB | 0.894 | 0.937 (+5%) | 0.528 | 0.559 (+6%) | 0.867 | 0.917 (+6%) |
| 5,000 | 23.6 GB | 0.885 | 0.922 (+4%) | 0.510 | 0.527 (+3%) | 0.853 | 0.892 (+5%) |
| 5,043 (H-Net end) | 23.8 GB | | 0.920 (+4.0%) | | 0.524 (+2.7%) | | 0.891 (+4.5%) |

Held-out bytes per chunk at the end: ClimbMix 4.38, code 4.13, math 3.88 (GLM:
4.83, 3.99 and 3.62 bytes per token).

Findings:

- At 5.2B-token-equivalent data and matched FLOPs per byte, the byte-level H-Net
  is still worse than the GLM-tokenized model on every source, by 2.7–4.5% in bits
  per byte.
- The gap shrank from about 20% at 2.4 GB to about 4% at 23.8 GB, most of that by
  3,000 steps; between 3,000 and 5,000 steps it moved little (ClimbMix +5% to +4%,
  code +2% to +3%, math +5% to +5%). These numbers do not show whether, or when, it
  would cross over within 40B tokens.
- It costs about 2.2x the GLM model's wall-clock time per byte here (850–890K vs
  about 1.69M bytes/s on 8 GPUs), because the byte-level KDA/SwiGLU layers run at
  about 10% MFU. At equal GPU-hours the H-Net would be further behind.
- Routing needed AdamW and a 10x stronger ratio loss than the paper's 0.03 to
  stay near the target (above). Cartesia's improved H-Net, which this is not,
  reports changes to exactly this mechanism.
- Greedy samples at the end are fluent but loose, similar in kind to the GLM
  model's samples at a similar point ("The history of the printing press is a long
  one. It is a time of innovation and change..."; `fibonacci` returns
  `n * (1 + n)`; the prime-number prompt loops).

## Samples

[`analysis/sampling/recipes/hnet-vs-glm-720m-samples.json`](../analysis/sampling/recipes/hnet-vs-glm-720m-samples.json)
samples the H-Net (23.8 GB) and the GLM model's 5.0B-token export (step 4,769, 22.7 GB
of the same data) on the toolkit's 20-prompt fluency set: greedy, and temperature
0.8 / top-p 0.95 with two seeds. Each sample is about 320 bytes (72 GLM tokens,
about 270 characters). Results, report and all completions:
`training/results/hnet-720m/samples/` in the results dataset.

| Model | Mode | Char16 first 128 | Char16 first 256 | Word4 first 32 |
| --- | --- | ---: | ---: | ---: |
| GLM hybrid | greedy | 19.1% | 26.0% | 20.7% |
| H-Net | greedy | 18.5% | 40.8% | 21.9% |
| GLM hybrid | t0.8 / p0.95 | 1.9% | 2.2% | 2.0% |
| H-Net | t0.8 / p0.95 | 2.2% | 5.7% | 2.7% |

- **Fluency:** both models produce fluent prose with confident factual errors.
  - The French Revolution began in "1532" (GLM) or "1930" (H-Net).
  - The H-Net gets "The capital of France is Paris"; the GLM model loops on "the capital of France".
- **Arithmetic and code:** neither model gets them right yet.
- **Repetition:** over the first 128 characters, about the same; over 256, the H-Net repeats more.
- **Prompts ending mid-token:** the byte model handles prompts that end in spaces. The GLM model continues `def factorial(n):\n    ` with a space-prefixed ` return`.

The 720M GLM hybrid remains the better model for this budget. Further H-Net work
would need faster byte-level layers (most of the wall-clock cost) and the improved
routing once Cartesia releases it, before a 40B-equivalent run is worth its
about $2,000 (about 58 hours on 8 GPUs) at the current throughput.
