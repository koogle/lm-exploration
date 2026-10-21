# Measured GPU comparison for 20B-token pretraining

> The measurements below used the recorded SDPA source snapshots. The modern
> model now uses the same causal SDPA implementation, introduced separately from
> the pilot and benchmark tooling in PR #10. Raw results and provenance are unchanged;
> these throughput estimates do not apply to the historical explicit-attention implementation.

Measured on Modal on 2026-10-17. The user requested hardware research before the
full run and prioritized elapsed time over GPU cost. **Three H100s are the
fastest measured option. No full 20B-token training run was launched.**

The historical `100m` model has 124,389,632 parameters with the frozen 32,768-entry
structured-code tokenizer. These measurements use the actual model and pilot
corpus, not theoretical FLOPs or synthetic-token inputs.

## Matched hardware comparison

Every trial uses context 2,048, 48 global document windows per optimizer update,
BF16 autocast with FP32 parameters, causal PyTorch SDPA, fused AdamW, gradient
clipping at 1.0 and the same seed/data prefix. Padding is excluded from both loss
and token counts. Increasing GPU count does **not** increase the effective batch.
One host runs NCCL DistributedDataParallel for two/three GPUs. Rank zero loads
and scatters the global batch; gradients are normalized by global loss-token
count, including unequal nonpadding counts across ranks.

We swept microbatch sizes with two warmup and six timed updates, then confirmed
the fastest configuration over 30 timed updates. Each confirmation processes
exactly 1,993,452 loss-bearing tokens and finishes at the same data cursor.

| Hardware | Loss tokens/s | Microbatch/GPU × accumulation | 20B compute hours | GPU-only cost estimate |
| --- | ---: | ---: | ---: | ---: |
| A10 (Modal `A10G`) | 22,853 | 8 × 6 | 243.1 | $268 |
| L40S | 57,257 | 4 × 12 | 97.0 | $189 |
| H100 | 131,439 | 24 × 2 | 42.3 | $167 |
| 2 × H100 | 249,626 | 24 × 1 | 22.3 | $176 |
| 3 × H100 | 337,293 | 16 × 1 | 16.5 | $195 |

Compute hours are `20e9 / measured_loss_tokens_per_second / 3600`. Pricing from
[Modal's pricing page](https://modal.com/pricing), retrieved 2026-10-17:
A10 $1.1016/hour, L40S $1.9512/hour, H100 $3.9492/GPU-hour. Costs exclude CPU,
host memory, storage, preparation, evaluation, checkpoints and restart overhead.
The user prioritizes speed; the cost column is context, not a selection constraint.

## Two/three L40S and A10 GPUs

The user subsequently requested the same comparison for two and three L40S/A10
GPUs (interpreting “a19s” as A10s from the preceding request). Each configuration
uses the same batch policy and a fresh microbatch sweep followed by **100 timed
updates**, without benchmark-buffer replay. H100's longer confirmation below uses
500 updates; single-device A10/L40S measurements above use 30. These durations are
explicit so short-run variation is not mistaken for exact scaling.

| Hardware | Loss tokens/s (100 updates) | Microbatch/GPU × accumulation | 20B compute hours | GPU-only cost |
| --- | ---: | ---: | ---: | ---: |
| 2 × L40S | 98,571 | 4 × 6 | 56.4 | $220 |
| 3 × L40S | 154,341 | 4 × 4 | 36.0 | $211 |
| 2 × A10G | 44,169 | 8 × 3 | 125.8 | $277 |
| 3 × A10G | 64,683 | 8 × 2 | 85.9 | $284 |

All four process 6,619,763 timed loss tokens with identical
data-cursor trajectory hash `ac5036cd3a12b6268de24a6fa4981154033d0ca39d762deabcfde075b8219f73`.
Their per-update token counts also match the first 100 updates of every H100
confirmation. Three L40S GPUs beat one H100, but remain slower than two H100s.
Even three A10s take about three and a half days of compute.


The raw results and trial logs are on the same results volume under
`simple-llm-124m-multigpu-la-20261017-01/{L40Sx2,L40Sx3,A10Gx2,A10Gx3}`;
[Modal app](https://modal.com/apps/radiantai/main/ap-lTq6ysbp1Q0jkzdt29RNES).
The checked-in files use the same run ID and hardware suffix.

## Longer H100 confirmation

A separate run confirms the selected microbatches for 500 timed updates each.
The pilot buffer is explicitly replayed for this bounded speed measurement;
no model weights are retained and this does not relax the production two-pass
policy. Data-cursor trajectory hashes and per-update token counts permit a
matched-input check across all three configurations.

| H100 count | Loss tokens/s (500 updates) | 20B compute hours | Speedup vs one |
| --- | ---: | ---: | ---: |
| 1 | 132,281 | 42.0 | 1.00× |
| 2 | 245,552 | 22.6 | 1.86× |
| 3 | 345,895 | 16.1 | 2.61× |

All three process 33,299,919 timed loss tokens with identical
per-update token counts and data-cursor trajectory hash
`4952313146318a2fcab31c76b358d4dde5864f35c6cb0f7da3b21611aa901a96`.
Each resets the benchmark buffer 4 times.


Budget **roughly 20–24 hours of training on three H100s** as an initial planning
allowance, subject to production-loader and checkpoint/evaluation measurements.
This is not an end-to-end completion promise: corpus preparation is separate and
has not been timed at full scale. The measurements last minutes rather than days,
use a small buffered corpus, and do not establish long-run reliability or quality.

The earlier 9–13 day L40S projection came from the original pilot with explicit
attention, context 1,024, smaller batches and frequent checkpoint/evaluation work.
The new table changes those settings as well as hardware. Its improvement must
not be attributed entirely to H100 hardware. The matched new L40S baseline itself
projects about four days of compute.

## Correctness and negative results

- 24 architecture, embedding and corpus unit tests passed on Modal before the
  hardware sweep. A new float64 test checks SDPA outputs and gradients against
  explicit causal attention; each single-GPU run also checks CUDA BF16 agreement.
- Two/three-GPU trials passed a tiny-model gradient and AdamW-update equivalence
  check against a single-model global batch with unequal masked-token counts.
- A10 microbatch 16, L40S microbatch 24 and H100 microbatch 48 exhausted memory.
  Two-A10 microbatch 12, three-A10 microbatch 16 and two-L40S microbatch 24
  also exhausted memory. These failures are preserved in sweep JSON. Selected batches fit without
  activation checkpointing. Smaller batches are not necessarily slower: L40S
  favored microbatch 4 in this sweep.
- Initial apps `ap-5eYxqQnWRaGfm1zrmzfAh9` and `ap-XabQLFtEXSDpL0BfK6TZcx`
  failed before benchmarking because an imported Modal launcher was not mounted
  in the image. They were stopped; the corrected image explicitly mounts it.
- The confirmation's single-GPU JSON inherits a 24-GiB host-memory metadata
  field from the sweep helper; its actual launcher requested 32 GiB, as did the
  multi-GPU confirmation. The launcher now overrides that field for future runs.

## Artifacts and reproduction

All run artifacts are on Modal volume `lm-exploration-training`. Checked-in JSON
contains metrics/configuration, tokenizer/manifest identities and source hashes;
no corpus text, credentials or model checkpoints are checked in.

- [Single-GPU sweep app](https://modal.com/apps/radiantai/main/ap-FLZw0mYDpmMPkEjPcVxwVe):
  `simple-llm-124m-gpu-comparison-20261017-02/{A10G,L40S,H100}/report.json`.
- [DDP sweep app](https://modal.com/apps/radiantai/main/ap-ro53yiAlMrnjl22R0rhO7W):
  same run ID under `H100x2` and `H100x3`.
- [500-update confirmation app](https://modal.com/apps/radiantai/main/ap-JVhobaWqL43D8E2TmrKqpu):
  `simple-llm-124m-h100-confirmation-20261017-01/H100x{1,2,3}/report.json`.
- Corresponding local JSON files use `<run-id>-<hardware>.json` in this directory.
  Git HEAD in provenance is the base commit; per-file hashes identify uncommitted
  source used for a run. Later documentation/launcher fixes do not rewrite results.

From `training/`, with Modal credentials and the existing pilot volume:

```sh
uv run --with 'modal[api-proxy-support]==1.6.2' modal run --detach modal_gpu_benchmark.py --run-id NEW_SINGLE_RUN_ID
uv run --with 'modal[api-proxy-support]==1.6.2' modal run --detach modal_multi_gpu_benchmark.py --run-id NEW_DDP_RUN_ID
uv run --with 'modal[api-proxy-support]==1.6.2' modal run --detach modal_gpu_confirmation.py --run-id NEW_CONFIRMATION_RUN_ID
uv run --with 'modal[api-proxy-support]==1.6.2' modal run --detach modal_multi_gpu_benchmark.py --run-id NEW_LA_RUN_ID --gpus L40S,A10G --sustained-steps 100
```

Each launcher has explicit timeouts, no retries and at most four containers (three in the H100 confirmation).
These commands run measurements, not full pretraining.

## Remaining production work

The plan requires capacity under at most two passes/source, parent-grouped splits,
deduplication and contamination audits. The pilot has 283.9M total unique train
loss tokens but only 1.21M code tokens, limiting the balanced mixture to about
8.08M exposures under the two-pass rule. Replaying it to 20B would violate the
plan. Full-scale eligible corpus preparation and audits remain necessary.

The DDP benchmark does not save checkpoints. Before substantial training, the
production launcher needs multi-GPU checkpoint/restart integration, durable
optimizer/RNG/data-cursor restoration, evaluation and a token-budget stop. The
existing pilot verifies checkpoint reload but is not that production launcher.
Hardware selection is now measured; these data and operational requirements
remain distinct from the GPU speed question.
