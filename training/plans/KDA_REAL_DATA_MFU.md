# Real-data MFU qualification — November 5

**Latest decision (November 6):** the qualified WY kernel and exact Muon CUDA-graph
replay are enabled on the supported Hopper path. Muon retains individual-matrix
arithmetic. Full pretraining remains held. All-rank step measurements and matched
qualifications are recorded below.

Full pretraining is paused at the user's request. Its production app was stopped
during corpus preparation, before any pretraining stage started. The separate
preparation-only workflow can finish the corpus but cannot allocate training GPUs.

The benchmark now reads the actual frozen 50/40/10 prepared mixture through the
production loader and shared training loop. These are bounded training-step
measurements on the beginning of the prepared streams, not full-run throughput:
they exclude evaluation, checkpointing and cold-start compilation. No benchmark
checkpoint is promoted into pretraining.

## Profiling and controlled measurements

The initial eight-GPU real-data profile assigned about 46% of GPU kernel time to
matrix multiplication, 21% to KDA, 12% to compiled elementwise work, and 7% to
communication. Full attention was about 3%. Kernel-time shares can include overlap
and are not additive wall-time savings estimates. MFU uses the existing
matmul/full-attention FLOP estimate and the 989 TFLOP/s BF16 peak for H100/H200;
it does not enumerate every KDA recurrence operation.

Modal fulfilled some H100 requests with H200s. Earlier separate-allocation results
did not record the delivered SKU, so they do not establish a controlled software
speedup. The updated benchmark records GPU model, UUID, memory and power limit.
The comparisons below run both variants sequentially within the same allocation,
reset model/data seeds, hold global batch and optimizer constant, exclude the
first five steps, and run without the profiler enabled.

| Hardware and comparison | Reference tokens/s | Candidate tokens/s | Reference → candidate MFU | Peak allocated GiB/GPU |
| --- | ---: | ---: | ---: | ---: |
| One H100, direct FP32 gradient accumulation | 41,617 | 41,741 | 30.74% → 30.83% | 55.24 → 50.07 |
| Eight H200s, direct FP32 gradient accumulation | 352,823 | 353,498 | 32.48% → 32.52% | 55.24 → 50.18 |
| One H200, microbatch 4 → 8, direct accumulation | 44,623 | 47,959 | 32.91% → 35.43% | 50.07 → 80.14 |
| One H100, default → GEMM autotuning | 39,964 | 40,392 | 29.48% → 29.85% | 50.07 → 50.12 |
| Eight H100s, recompute → save KDA intermediates | 332,183 | 338,534 | 30.57% → 31.12% | 50.07 → 57.92 |

The custom operation saves approximately 9% of peak allocated memory. Its measured
0.2–0.3% throughput difference is within likely run-to-run noise: **no meaningful
speedup has been established**. The larger microbatch yields 7.5% more throughput
on H200, but requires more memory than an 80GB H100 provides and has not been
qualified across eight GPUs. These results are not a drastic MFU improvement.

Increasing loss chunks from 2048 to 8192 or 16384 did not improve the exploratory
real-data timings. Those independent allocations lack complete hardware identity;
they are negative exploratory evidence rather than a precise ranking. Production
keeps 2048. The completed same-allocation GEMM-autotuning comparison gained only
1.1% throughput over 19 timed steps, with substantial additional compilation
time. Production retains the default compilation mode. The short measurement
does not establish a reliable advantage at that margin.

## Custom operation and correctness

`hybrid/output_gradient.py` and `hybrid/kernels/output_gradient.cu` register a
CUDA operation that accumulates `grad_logits.T @ hidden` directly into an FP32
buffer using cuBLAS BF16 tensor-core GEMM with beta=1. This avoids a full
vocabulary-by-width FP32 temporary and a separate add for each loss chunk.
The model dimensions, objective, precision and optimizer remain unchanged.

CUDA tests check repeated accumulation from a nonzero buffer, irregular shapes,
the actual 154,880-by-1,536 output dimensions, and compiled execution. The tested
accumulations exactly matched the reference. The full fused loss matched the
autograd loss values; relative maximum gradient differences were 0.116% for hidden
states and 0.264% for weights, inside the existing BF16 tolerance. Matched
30-step runs stayed finite; final main losses differed by less than 0.00024.
This establishes implementation agreement over these checks, not long-run quality.

The output-gradient reference implementation is selectable in the benchmark for
controlled comparisons. Production uses the tested direct-accumulation operation for its
memory benefit. Its Python and CUDA sources are included in checkpoint provenance.

Every GPU call has a 30-minute server timeout, zero retries and an independent
launch-deadline supervisor. Completed comparison cases are persisted separately so
a failed later case does not discard them. Benchmarks fail if the requested step
count is not reached. The main run stays stopped while the performance decision
is reviewed; the preparation-only run has no automatic training continuation.

[Raw profiles, measurements and CUDA checks](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105)
preserve the exploratory runs as well as the matched comparisons.

## Next performance decision

The user has fixed the training target at **eight 80GB H100s**. Further
qualification requires the delivered GPU model to match H100 on every rank;
Modal upgrades to H200 are rejected before training, with hardware evidence
persisted. The H200 eight-GPU microbatch and one-GPU saved-KDA follow-ups were
stopped when this target was specified. Historical H200 measurements remain
archived but do not qualify the H100 training configuration.

Keep the full run held: the measured changes do not meet the requested substantial
MFU improvement. Preserve KDA, the frozen mixture and the training objective in
performance comparisons. A higher MFU number obtained by changing the FLOP
accounting is not a throughput improvement.

The next custom-kernel target should be selected from the KDA backward profile:
KDA accounts for about one fifth of kernel time, with backward intra-chunk and
fused backward kernels among the largest entries. A candidate needs a local
forward/backward reference comparison, then an unprofiled, same-GPU real-data
comparison at fixed global batch. Kernel-time reductions must translate into
end-to-end tokens/s before promotion. The profile alone does not establish that
rewriting these already specialized kernels will help.

Retain microbatch 4 for H100 qualification: the measured microbatch-8 allocation
exceeds H100 memory. Test saving KDA intermediates at microbatch 4, which trades
additional activation memory for less recomputation without changing the
recurrence or training objective. A successful CUDA reference check and a
same-allocation eight-H100 comparison are required before promotion. Keep checkpoint/evaluation overhead and
compilation cost separate from steady-state throughput when budgeting the full
run. Precision changes need their own loss/gradient and short-training checks;
BF16 results do not qualify FP8 training.

## Follow-up: where step time goes

The prior matched H100 run's median step was 3.140 s, with 1.20 ms of data fetch
and 2.63 ms outside its compute interval. The prior eight-H200 run's median step
was 2.966 s, with 6.85 ms of fetch, 9.47 ms outside the slowest rank's compute
interval, and a 2.94 ms spread between rank compute times. These intervals
include optimizer work, collective waits and CPU stalls inside computation;
they do not establish that the GPU is continuously busy. They make data loading
and gross rank imbalance unlikely primary bottlenecks. Logging/control work
outside the step timer still needs the full timeline.

The source audit identifies three concrete costs to separate:

- **Repeated document setup:** each of the 18 KDA layers rebuilds the same packed
  document offsets with CUDA `nonzero`. FLA then prepares variable-length chunk
  indices using `repeat_interleave`, which reads the output size on the host.
  FLA caches by tensor identity, but newly constructed offsets cannot share that
  cached result across layers. Sharing immutable per-microbatch offsets is a
  candidate; its wall-time saving is not yet measured.
- **Optimizer dispatch:** 255 Muon matrices each perform 15 Newton–Schulz GEMMs,
  giving 3,825 eager matrix multiplications per optimizer step. Their estimated
  arithmetic cost is 27.67 TFLOPs per rank per step. Grouping equal shapes or
  compiling the update could reduce dispatch/elementwise overhead, but the old
  kernel categories combine these GEMMs with model GEMMs, so their time must be
  measured separately before choosing an implementation.
- **Output projections:** the 154,880-token vocabulary has a 237.9M-parameter
  tied output table, used for both main and full-fraction MTP losses. These two
  uses account for approximately 39.7% of the matmul FLOPs in the current model
  estimate. The fused loss computes its output-layer gradients during forward,
  so a simple forward/backward split does not isolate this cost. Changing the
  vocabulary or MTP objective is not a kernel optimization.

The shared loop now labels forward/loss, local backward, final backward with
DDP synchronization, fetch, scatter/prepare, clipping/metric reductions,
optimizer, device wait and logging. The profiler records operator shapes, CPU
self time, GPU kernel time, the union of CUDA activity intervals and a compressed
Chrome trace. Nested phase times overlap; CUDA kernel-time sums are not wall-time
percentages. Shape recording adds overhead, so these traces diagnose causes;
unprofiled same-allocation comparisons establish speedups.

Saving KDA intermediates passed the existing independent recurrence reference
checks on an actual H100 80GB HBM3. Maximum relative errors across output and all
input/parameter gradients were 0.653% for FP32 inputs (FLA uses TF32 dots) and
0.704% for BF16. This qualifies the tested outputs/gradients, not throughput,
full-size memory fit, or long-run quality. The matched eight-H100 comparison completed 30 steps per arm (25 timed).
Saving intermediates gained 1.9% throughput but added 7.84 GiB of peak allocated
memory per GPU. Final main losses differed by 0.000244. This modest gain does
not justify adopting it as the leading optimization; production retains default
recomputation.


The first detailed trace is from one actual H100 with the production per-rank
batch (32 rows, microbatch 4); the eight-H100 trace is a separate pending run.
During its three instrumented steps, forward/loss took 1.528 s of host time,
backward 1.670 s, optimizer 0.239 s, and clipping/metric reductions 0.055 s.
The GPU kernel-time categories were 1.352 s GEMM, 0.675 s KDA, 0.431 s compiled
elementwise work, and about 0.09 s each for KDA convolutions, full attention and
loss softmax. These category times and host phase times are different views,
not quantities to add together.

The raw trace confirms 144 `aten::nonzero` calls per step, with 528 ms of
inclusive host time, and repeated chunk-index expansion. This includes time
waiting for useful preceding GPU work; removing the calls does not imply saving
528 ms. The optimizer's host span is 239 ms, with only about 97 ms of CUDA
activity during that span. This makes batching same-shape Muon matrices a
concrete follow-up candidate, not an established speedup.

The observed CUDA activity union covered 83.9% of the 3.508 s instrumented step
window. It is not MFU, and its complement is not a 16.1% guaranteed optimization:
the trace records 254 ms/step of command-buffer-full overhead that can be induced
by profiling. Six unprofiled tail steps measured 40.4K tokens/s and 29.8% MFU;
this short tail is diagnostic context, not a replacement for matched comparisons.
The original profiler summary did not recognize normalized `ProfilerStep*` names;
the separate `timeline-audit.json` derives activity intervals from raw Chrome
trace events. The reader now recognizes those normalized names and avoids adding
GPU annotation ranges to CPU phase attribution.

The shared-offset prototype and its boundary-isolation checks were evaluated
in the completed comparisons below. The active path uses kernel-local offsets.


## November 6: shared offsets and batched Muon qualification

The completed shared-offset comparison ran on eight verified H100 80GB HBM3s.
Within that allocation, per-layer offset reconstruction measured 302,081 tokens/s
and 27.78% MFU; shared offsets measured 313,340 tokens/s and 28.84% MFU: a 3.7%
throughput gain. Both completed 30 steps with 25 timed, identical first main loss,
and final main losses differing by 0.000036. This comparison reconstructed offsets
in the mask property; the follow-up restores reconstruction inside `kda_kernel`,
at its original call site, to control for compiler-boundary changes. Cross-host
throughput differences are not used as evidence of a software speedup.

The completed eight-H100 diagnostic confirms the repeated 144 `nonzero` calls per
step and an eager optimizer host phase around 176 ms, with roughly 96 ms of GPU
activity during that phase. NCCL kernel times across the three profiled steps were
40, 562 and 42 ms; the large spike makes the average unreliable as a steady-state
communication estimate. Communication settings remain unchanged.

The evaluated Muon prototype grouped equal-shape, equal-dtype, same-device
matrices within each parameter group, retaining independent normalization,
momentum and aspect-ratio scaling. It tested batches of up to 16. The active
optimizer processes matrices individually; the prototype source is preserved
in Git history and its measurements in the HF archive.

CPU tests cover different shapes and learning rates, zero and absent gradients,
partial batches, momentum agreement and state restore. H100 CUDA checks cover the
production matrix dimensions, a full batch of 16 and partial batches over four
steps. Tested updates and momentum buffers match the individual-matrix reference
exactly. This is a correctness result, not a speed or long-run training claim.

The completed bounded eight-H100 comparison ran four sequential arms on the same
allocation: original kernel-local document setup, shared offsets, shared offsets
plus batched Muon, then the original setup again. All use the frozen real mixture,
context 4096, global rows 256, microbatch 4 and 30 steps. The final repeated
baseline checks for clock/warm-up drift. Per-case outputs persist immediately;
server timeout remains 30 minutes, retries zero, with an independent launch
supervisor. Its completed outcome is recorded below; full pretraining remains held.


## Final four-arm qualification: candidate rejected

All four arms completed on the same eight verified H100 80GB HBM3 devices, with
30 steps each and 25 timed steps per arm. No profiler was active.

| Variant | Tokens/s | MFU | Peak allocated GiB/GPU |
| --- | ---: | ---: | ---: |
| Original, first | 327,665 | 30.18% | 50.07 |
| Shared offsets | 312,322 | 28.77% | 55.24 |
| Shared offsets + Muon batch 16 | 314,321 | 28.92% | 55.24 |
| Original, repeated | 328,127 | 30.23% | 50.18 |

The repeated original differs by only +0.14%. Shared offsets regress throughput
by 4.68%; the combined candidate regresses by 4.07%, relative to the first
original. Batched Muon adds only 0.64% over the shared-offset arm, too small to
establish a meaningful gain. Neither candidate is adopted. The default model
uses original kernel-local offset reconstruction, and the default optimizer
processes one matrix at a time. Historical prototype sources and measurements remain available in Git history
and the HF archive; active training and benchmark defaults use the original path.

The earlier +3.7% result compared shared offsets with per-layer reconstruction
in the mask property. That control moved work out of its original kernel call
site and did not establish an improvement over the original implementation.
The faithful original-path comparison supersedes that apparent overall gain.
Compiler behavior is a plausible reason for the slowdown and approximately
5.2 GiB additional peak memory, but the responsible generated kernels have not
been isolated. Correctness checks and finite short-run losses did not predict
performance: final main losses across these arms differ by less than 0.0004.

Raw per-case logs, arguments, hardware identities and the repeated baseline are
preserved in the linked HF archive. The evidence supports keeping the original
path, not launching full pretraining on the slower candidate.


## MFU diagnosis and the requested 40–60% target

The user requests findings and the original training path, with rejected
shared-offset and batched-Muon prototypes retained as historical evidence.
This analysis reuses the completed eight-H100 trace; no new GPU optimization
runs are required for these findings.

### The size of the gap

The original-path median step is 3.200 s for 1,048,576 global tokens. The logged
MFU formula implies about 7.279 GFLOPs/token for this mixture and model, including
both main and MTP output projections and estimated full attention. Its denominator
is eight times the **dense** H100 BF16 peak of 989 TFLOP/s, not the sparse peak.
Attention lengths are sampled on rank zero; KDA recurrence, optimizer arithmetic,
normalization and softmax are not comprehensively counted as model FLOPs. This
is a model-FLOP efficiency estimate, not a hardware activity percentage. Changing
the accounting would not make training faster.

| Target MFU | Required step time | Required tokens/s | Step time to recover |
| --- | ---: | ---: | ---: |
| 40% | 2.412 s | 434,807 | 0.789 s |
| 50% | 1.929 s | 543,508 | 1.271 s |
| 60% | 1.608 s | 652,210 | 1.592 s |

These targets hold the current model, objective, context and global batch fixed.
The trace's CUDA activity union is 3.077 s in a 3.470 s instrumented window
(88.7% activity), with profiler-induced stalls. It does not support the claim
that 70% of wall time is idle. Nor is its roughly 11% inactive interval a measured
recoverable fraction of unprofiled runtime.

### What the kernels reveal

- **Dense GEMMs are relatively efficient.** The arithmetic model gives about
  941.6 TFLOPs of dense model GEMMs per rank per step, plus 27.7 TFLOPs for Muon's
  Newton–Schulz work. Dividing by the trace's 1.375 s GEMM kernel-time sum gives
  roughly 705 TFLOP/s, or 71% of nominal peak. This is an approximate aggregate
  estimate, not a tensor-pipe counter. Even perfect GEMM execution would recover
  only about 0.40 s under this accounting, short of the 0.79 s needed for 40% MFU.
- **KDA has a substantial non-GEMM cost.** KDA kernels consume about 0.673 s per
  profiled step. Two backward kernels account for 0.404 s of that. Their launch
  metadata records 254 and 255 registers/thread. With 64K 32-bit registers/SM,
  their allocations permit at most eight resident warps per SM: approximately
  12.5% of the 64-warp limit. This is a concrete register-pressure constraint,
  not measured achieved occupancy or proof of register spilling. Dense GEMMs
  can also work well at low warp occupancy through different execution pipelines.
- **Elementwise work is material.** Compiled elementwise kernels consume about
  0.432 s. Examples include dtype copies (one named kernel totals 53 ms and
  2,560 launches/step); eager addition kernels add another 102 ms in the separate
  `other` category. FLOP-based MFU gives little credit to these operations even
  when they keep execution units and memory paths busy.
- **The output objective moves substantial data.** Main plus full-fraction MTP
  loss processes 262,144 token positions per rank through a 154,880-token
  vocabulary. The softmax-gradient kernel reads logits twice and writes them
  once: about 243.6 GB of logical loads/stores per step. Its 85.8 ms kernel time
  implies 2.84 TB/s effective logical bandwidth. Caches may serve some traffic;
  this is not a measurement of DRAM bandwidth. The number is consistent with
  substantial bandwidth work, and the entire kernel is under 3% of step time.
- **Data loading and stable communication are smaller targets.** Fetch is a few
  milliseconds. NCCL kernel times in the three trace steps are 40/562/42 ms;
  the isolated spike prevents interpreting their average as normal collective
  cost. CPU wait durations include waiting for useful GPU execution and cannot
  be counted again as free time to eliminate.

### What is established, and what is still unknown

The gap is largely a mixture of high-throughput GEMMs and much slower-per-FLOP
recurrence, elementwise, normalization and data-movement work. Register-heavy KDA
backward kernels are an evidence-backed focus for deeper diagnosis; the failed
setup/optimizer prototypes did not address their main cost. The 40% target
requires substantial reductions in these kernel costs, not just host cleanup.
The evidence does not establish that 40% or 60% is achievable for this unchanged
workload, and 60% would require nearly halving the current step time.

The trace does **not** contain measured DRAM/L2 bandwidth, tensor-pipe utilization,
SM clocks, warp-stall reasons or local-memory spill traffic. Consequently it
cannot distinguish register-induced latency from memory or instruction-pipeline
limits inside KDA conclusively. Its generic occupancy estimator even reports
zero for some executing high-shared-memory GEMMs, so those automatic estimates
are not treated as measured hardware utilization. Kernel resource metadata and
arithmetic/traffic budgets are archived with these explicit limitations. Hardware
counter profiling, rather than another speculative optimization, is the missing
evidence needed for a firmer kernel-level root cause.

### Hardware-counter access (2026-11-06)

Nsight Compute 2025.3.1 can collect hardware counters on the tested Modal H100
80GB HBM3 hosts (driver 610.57.04). The initial probe failed because Nsight's
default clock locking was rejected by the host. Setting `--clock-control none`
allowed actual `sm__cycles_elapsed.avg` and `dram__bytes_read.sum` measurements;
this was not a counter-permission denial. Both attempts are preserved in the
[HF evidence archive](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105),
under `counter-access-h100` and `counter-access-h100-02`.

The existing benchmark now supports bounded Nsight collection after six warmup
steps. An attempted eight-H100 kernel replay failed with NVSwitch SXid 12028,
GPU Xids 137/94 and peer-memory errors when replay began; it produced no valid
training-kernel counters. That attempt also used an older study recipe with 2K
windows, caught from its actual token counts; it is setup evidence, not a 4K
qualification. The benchmark now rejects recipe/model context mismatches.

Kernel replay is restricted to one H100. The isolated diagnostic uses the
foundation mixture, context 4096, microbatch 4 and global batch 32 (the eight-GPU
run's per-rank workload). The data cursor differs from rank zero of the full
eight-GPU job. These counters cannot measure DDP contention or replace the
eight-GPU timing trace. A selected kernel invocation is replayed for
compute/memory throughput, occupancy, scheduler and warp-stall sections. Raw
reports and CSV metrics accompany the run logs. Replayed timings are explicitly
excluded from throughput/MFU summaries. Provider clocks and caches are unmanaged;
replay passes may observe different cache states and clocks. These are sampled
kernel diagnostics, not a whole-run or whole-cluster utilization measurement.

### Measured KDA hardware counters

The two isolated-H100 runs completed all eight training steps with finite losses
and generated valid Nsight reports. Each sampled one warmed-up kernel invocation.
The raw-page exporter uses metric columns plus a units row; the initial validator
expected the probe's long-form CSV and falsely flagged the exports as empty. The
reader is corrected, and `counter-audit.json` validates the existing reports
without another GPU run. Original result files and raw reports remain preserved
under `counter-intra-h100` and `counter-wy-h100` in the HF archive above.

| Measured counter | Intra backward | Fused WY/dqkg backward |
| --- | ---: | ---: |
| Kernel duration | 1.372 ms | 1.368 ms |
| Registers per thread | 254 | 255 |
| Achieved warp occupancy | 12.02% | 12.47% |
| Tensor-pipe active cycles / elapsed cycles | 1.07% | 2.71% |
| Scheduler issue-active cycles / active cycles | 39.92% | 7.26% |
| Eligible warps per scheduler per active cycle | 0.491 | 0.076 |
| DRAM bandwidth | 0.683 TB/s | 2.002 TB/s |
| DRAM throughput / peak sustained | 20.37% | 59.74% |
| L1 load/store wavefront throughput / peak sustained | 79.57% | 19.31% |
| Nsight-derived local spilling requests | 0 | 13,266,960 |

The fused WY backward kernel has a concrete memory/dependency bottleneck: almost
no warps are eligible to issue, long-scoreboard stalls dominate the reported stall
categories, and Nsight reports substantial local/spill requests. Its measured
DRAM and L2 throughput are both about 60% of peak, while tensor-pipe activity is
only 2.7%. These counters support investigating live tensor/register pressure and
spill traffic. They do not provide a byte-accurate attribution of DRAM traffic to
spilling: the SASS-level spill-byte metric is unavailable in these reports.

The intra backward kernel behaves differently: no derived spill requests, about
80% L1 load/store wavefront throughput, only 20% DRAM throughput, and substantial
long- and short-scoreboard stalls. This points toward on-chip load/store work and
latency hiding, rather than saturated HBM bandwidth. Its 12% achieved occupancy
confirms the earlier register-capacity concern, but low occupancy alone does not
identify an implementation fix.

At the sampled launch configurations, registers cap intra at eight blocks per SM
and fused WY at two. Nsight also reports a two-block shared-memory limit for WY
at its selected 135.168 KB shared-memory carveout. That carveout can change with
kernel configuration, so lowering registers must be measured rather than assumed
to raise occupancy. Stall ratios in the raw CSV are per issue-active units, not
percentages of whole-step wall time.

The sampled durations are close to the earlier eight-GPU trace's averages
(approximately 1.43 and 1.37 ms per call), but the invocation/document layout and
profiling method differ. These are isolated-kernel findings; they do not resolve
DDP communication stalls or measure all eight devices concurrently.

Together these two kernels cost about 0.404 seconds in the eight-GPU timeline.
Even halving both would recover only about 0.20 seconds, assuming that saving
translates fully to the critical path. Applied to the 3.20-second baseline, that
would yield roughly 32% MFU, not 40%. The requested 40% still needs about 0.79
seconds of total step-time recovery. The counters establish specific sources of
inefficiency; they do not establish that 40–60% is attainable with this workload.
The resulting WY optimization and its qualification are described below.

### WY backward implementation change

The H100 specialization in `hybrid/kernels/kda_wy.py` moves the value-only
`dv2`/`dA`/`db` work before the three key-gradient accumulators become live. It
uses key/value tiles 32/128, eight warps and one pipeline stage. The mathematical
operation and returned gradients remain the same; the scheduling trades an extra
input read for shorter live ranges. `hybrid/kda_wy.py` connects it through FLA's
operation-specific backend registration for H100, head dimension 128 and chunks
of 64. The source is adapted from the pinned FLA 0.6.0 implementation under MIT.

On one actual H100, using inputs produced from the foundation recipe's packed
4K windows, the original kernel measured 1.3678 ms and the revised kernel
0.6755 ms (median of three CUDA-graph timing batches): 50.6% less kernel time.
Compiler-reported spills fell from 360 to zero. A layout-only change also helped
(1.3635 to 0.7392 ms with zero spills), so both layout and live ranges matter.
The six output buffers passed the original-kernel comparison; the largest
normalized max difference for the selected layout was 0.00455 in the gate
gradient, reflecting the changed value-tile accumulation order.

Full KDA forward and all seven input gradients passed the FP64 recurrence
reference on packed lengths 127, 193 and 257 in BF16 and FP32. The test confirmed
six custom-backward invocations. Maximum normalized errors were 0.00755 in BF16
and 0.00199 in FP32; at the shared length-193 case the recorded maximum errors
matched the original implementation's. These are normalized max errors, not
per-element relative-error guarantees.

Raw layout comparisons and full gradient checks are under `wy-fix` in the
[HF archive](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/wy-fix).
The same-allocation eight-H100 qualification completed original, revised and
original-again arms using 30 steps each, the real foundation recipe, global batch
256, microbatch four and 4K windows. Each median excludes five warmup steps.

| Arm | Tokens/second | MFU |
| --- | ---: | ---: |
| Original first | 329,964 | 30.41% |
| Revised WY | 339,397 | 31.23% |
| Original repeat | 329,521 | 30.33% |

The revised kernel improves whole-training throughput by 2.86–3.00%; the two
controls differ by only 0.13%. All 90 steps have finite losses. Across matched
steps, the maximum main-loss difference is 0.000479 and MTP-loss difference is
0.000163. Candidate peak allocation is 50.18 GiB per GPU versus 50.07 GiB for
the first control. These short runs qualify numerical stability and throughput,
not long-run training quality.

The specialization is now registered by the hybrid family on CUDA; dispatch
selects it for compute capability 9.0, BF16/FP32, head dimension 128 and chunks
of 64. H100 is the measured hardware. Checkpoint source provenance includes both
the launcher and kernel. The benchmark's `--compare-kda-wy` option runs the same
original/revised/original comparison using separate processes; reference arms
suppress this registration. All 32 backbone, target-model, workflow and deadline
CPU tests pass, and full KDA gradient checks pass on GPU.

Whole-training MFU remains about 31.2%, below the requested 40–60%. The isolated
kernel's 50.6% time reduction must not be presented as a whole-step reduction.
Full pretraining remains held. Matched results and logs are in the archive's
`wy-fix/matched-h100x8` folder.

### Accounting for elapsed time and testing optimizer launch overhead

The earlier three-category summary was incomplete: subtracting GEMM, KDA and
compiled-elementwise kernel sums from an unprofiled wall-clock step does not
produce a valid estimate of unexplained or idle time. The archived trace also
contains approximately 215 ms of communication, 207 ms of other kernels, 91 ms
of KDA convolutions, 87 ms of full attention, 86 ms of loss softmax/gradient work,
9 ms of memory operations and 6 ms classified by optimizer kernel names. These
categories sum across streams; they are distinct from exclusive wall-clock
phases. Optimizer GEMMs and additions appear in the GEMM and other categories.

The trace reader now correlates each device operation with its CPU launch and
chooses the innermost enclosing training phase. On the earlier trace, backward
launches account for about 1,854 ms of summed kernel time; forward/loss launches
account for 1,223 ms. The clipping/metrics phase launches only 4.1 ms of device
work despite its 176 ms host interval: the host also waits for delayed backward
work. Optimizer launches account for 97.6 ms of kernel work within a 176 ms host
interval. This is a concrete reason to measure optimizer submission overhead,
not evidence that the entire host interval is recoverable. The trace includes
an anomalous communication event and remains diagnostic rather than a normal-step
latency measurement.

`--phase-timings` adds CUDA-event and host timings on every rank without new
synchronization barriers. CUDA-event intervals include stream waits and host
submission gaps. Parent `accumulate` and child forward/backward intervals overlap.
These measurements complement the kernel trace; they do not measure tensor-core
utilization or expose every NCCL stream separately.

Ordinary compilation of the Muon orthogonalization was rejected by its numerical
gate: even with precision-cast emulation, the first production shape differed by
3.23% normalized maximum and 1.39% relative RMS. The subsequent CUDA-graph candidate
replays the existing operation sequence and passes bit-for-bit checks on all seven
production matrix shapes, with three different inputs per shape. Its graph output
is a borrowed buffer, consumed by the parameter update before the next replay.
The candidate keeps individual-matrix Muon arithmetic and the checkpoint state
layout. A queued microbatch-eight retest was cancelled before GPU allocation:
existing measurements already show that its memory requirement exceeds H100 capacity.


The same-allocation graph qualification completed 30 steps per arm on eight H100
80GB GPUs, using the real mixture, 4K context, global batch 256 and microbatch four.
All arms use lightweight event timing; the first five steps are excluded.

| Arm | Tokens/second | MFU | Rank-zero optimizer stream interval |
| --- | ---: | ---: | ---: |
| Original first | 335,055 | 30.87% | 114.5 ms |
| Muon graph replay | 337,132 | 31.03% | 104.8 ms |
| Original repeat | 335,973 | 30.91% | 116.2 ms |

The whole-step gain is modest: 0.35–0.62%, versus 0.27% control drift. The reduced
optimizer interval supports a small local improvement, not a large utilization
claim. Its host submission interval falls from 104.4 to 51.8 ms, but much of
that host work already overlapped device execution. Candidate peak allocation is 50.51 GiB/GPU; maximum matched main/MTP loss
differences are 0.000216/0.000362. All 90 steps finish with finite losses. Default
Muon dispatch selects graph replay for FP32 parameters on compute capability 9.0;
H100 is the qualified hardware. Graph buffers are rebuilt on resume; the saved
momentum/parameter-group state layout is unchanged.

The first control's median wall step is 3.130 s. Rank-zero CUDA-stream intervals
are approximately 2.994 s for accumulated forward/backward, 114.5 ms for the
optimizer, 7.1 ms for fetch, 0.8 ms for preparation and 4.3 ms for clipping/metrics.
Medians need not add exactly. Forward/backward includes dependency waits and
communication; it is not a kernel-busy measurement. Across ranks, the accumulated
interval agrees closely, while faster local work can shift waiting into the final
synchronized backward. Feeding data is not the dominant steady-state bottleneck.

Raw logs, all-rank phase medians, numerical checks, the rejected compilation and
cancelled queued test are in the archive's `phase-graph` folder. The graph gain
is measured against controls from its own allocation, not the earlier WY run.

The follow-up microbatch-six qualification keeps global rows at 256 and uses
`[6, 6, 6, 6, 6, 2]` on each rank. The shared loop allows the smaller final chunk
only for adapters that declare support; workflow validation limits this hybrid
path to the non-looped model. CPU tests confirm that averaging two ranks' NTP/MTP
gradients with masked targets matches the unsplit global objective. All 52
regression tests pass, including the other model families' validation boundaries.

The bounded comparison is queued as
`kda-phase-microbatch6-h100x8-20261106-01`
([Modal app](https://modal.com/apps/radiantai/main/ap-bekBaYIESy9BEtSHchpikY)).
It requests exactly eight H100s, 30 steps each for four/six/four microbatches,
with server-side execution timeout and independent launch deadline. Its per-arm
logs and results are persisted under `throughput/<run-id>` on the hybrid runs
volume. As of November 6, 13:48 UTC it has not received a GPU allocation; no
throughput, memory or training-loss result is established for microbatch six.
The production recipe remains at four pending that qualification.
