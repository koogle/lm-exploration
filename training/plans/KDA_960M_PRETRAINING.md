# KDA foundation pretraining — November 5

The user authorized pretraining with KDA and the tested 50/40/10 mixture, and
asked to measure and improve throughput/MFU before committing the main budget.
The 128M study's middle mix improved code BPB by 11.7% at a 3.8% general-text
regression; this explicitly exceeds the proposed 2% retention guardrail.

## Target and throughput

Use the implemented 24-layer, width-1536 KDA/full-attention 3:1 architecture,
4K context, GLM tokenizer and MTP objective: 959,843,116 training parameters,
927,589,920 deployed trunk parameters. Preserve the reasoning SFT/RL budget.

The first throughput comparison used the real shared training step on synthetic
packed documents (mean length 1,000), one H100, microbatch 2 and 16 global rows.
All arms used the same model, optimizer, seed and objective. Medians below use
steps 10–14 and 19–20, excluding compile/warmup and profiler-active steps. The
profiler initially repeated; its schedule now records one cycle only.

| Configuration | Tokens/s | Estimated MFU | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| Eager, loss chunk 128 | 15,868 | 11.7% | 44.6 |
| Eager, loss chunk 2048 | 24,862 | 18.3% | 45.1 |
| Compiled, loss chunk 2048 | 33,445 | 24.7% | 36.7 |

Larger loss chunks and compilation yield 2.1x throughput with lower memory.
Compiled kernel time is approximately 43% matrix multiplication, 21% KDA,
16% compiled elementwise and 3% full attention. Data fetch is about 1–2 ms/step.
MFU uses the repository's matmul-plus-full-attention FLOP estimate and 989 TFLOP/s
per H100; it is not a hardware-utilization reading and omits detailed KDA
recurrence FLOPs. These synthetic timings exclude evaluation, checkpoints,
startup and corpus I/O.

The subsequent unprofiled microbatch-4 benchmarks completed 20 steps each.
Steps 6–20 measured 39,814 tokens/s and 29.3% MFU on one H100 (32 global rows),
and 327,369 tokens/s and 30.1% MFU on eight H100s (256 global rows). Peak allocated
memory was 55.24 GiB/GPU. The slightly superlinear ratio is a short-run hardware/
timing observation, not an expected scaling guarantee. Select microbatch 4,
loss chunks 2048 and compilation for the production workflow.

[Raw results and profiles](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/throughput-960m-20261105)
preserve every arm, configuration and the timing-step selection. Every GPU call
has a 30-minute server timeout, zero retries and an independent deadline supervisor.

## Corpus and launch constraints

`hybrid/recipes/kda-foundation-100b-504010-v1.json` prepares a single-pass corpus
for a conditional 100B-token schedule: 50B ClimbMix, 40B permissive GitHub and
10B Nemotron CC-Math. Math remains the tested **score-3 subset**, not combined
3+ or 4+. The manifest pins upstream revisions; capacity estimates only size
preparation. The launcher now checks actual token counts and shard files before
allocating GPUs.

Preparation uses eight CPU workers, existing workflow supervision and the same
13-token benchmark screen including all six LiveCodeBench releases. Content-hash
partitions exclude exact train/validation overlap across sources; GitHub also
requires repository-hash separation. Duplicate normalized documents are dropped
within each preparation shard. Math files are partitioned by Parquet row groups
to bound worker memory. Global training deduplication, fork-aware near-deduplication
and generated/vendor-code filtering are not established by these checks; keep
this limitation in downstream benchmark interpretation.

`workflows/kda-960m-foundation.json` owns the production launch. Its data stage
prepares the frozen corpus with eight CPU workers; its eight-H100 training stage
uses `hybrid/configs/kda-960m-100b.json`. The large job must wait for preparation and capacity validation. Qualification
uses the final model/schedule, then resumes the same checkpoint lineage. Save
every 30 minutes, retain numerical held-out evaluation, and disable preview generation
to avoid the study's changing-shape compilation issue. No mid-training, SFT or RL
stage is launched automatically. Each launch remains independently bounded below
24 hours; continuation requires checking actual throughput, spend and progress.
GPU sessions have 21,600 seconds of training time, a 22,800-second server timeout
and zero retries. The first session stops after eight qualification steps. The
independent supervisor bounds all children, including preparation, below 24 hours.
Private inference exports go to `jfrick/kda-960m-504010-100b`; optimizer checkpoints
remain on `lm-exploration-hybrid-runs`.

The $5,000 main-pretraining allocation remains the ceiling. Modal's published
H100 rate checked November 5 is $0.001097/GPU-second ($3.9492/hour); CPU, memory,
preparation, storage and checkpoint/evaluation overhead must also be counted.
Set the final token commitment from measured scaling and real-data throughput,
not ideal eight-GPU linear extrapolation. The measured synthetic eight-GPU rate
projects 84.9 hours and approximately $2,681 in GPU charges for 100B tokens before
overheads. This leaves room within the allocation, but is not a billing guarantee.
Reconcile actual spend and real-data throughput before each subsequent launch;
do not start post-training from the reserved budget automatically.

The initial data-controller attempts failed before shard preparation: missing
NumPy, then a local path used instead of its resolved container path. Both are
fixed in the shared workflow; failed launch records remain preserved.
