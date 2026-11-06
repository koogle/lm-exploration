# Real-data MFU qualification — November 5

**Latest decision (November 6):** the four-arm same-allocation H100 test rejects
shared offsets and the combined shared-offset/batched-Muon candidate. The original
kernel-local setup is restored as the default; Muon remains unbatched. Full
pretraining remains held. See the final qualification table below.

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

The reference implementation is selectable only in the benchmark for controlled
comparisons. Production uses the tested direct-accumulation operation for its
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

Shared immutable document offsets are implemented in `DocumentMask` and reused
by all KDA layers in each microbatch. A test covers EOS boundaries, row boundaries,
and isolation between different batches. The benchmark can restore per-layer
reconstruction for a matched reference. Its eight-H100 comparison is pending;
full training remains held and no speedup is claimed for this change yet.


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

Muon now supports bounded batches of equal-shape, equal-dtype, same-device
matrices within each parameter group. Every matrix keeps its own normalization,
momentum buffer and aspect-ratio scaling. The default remains one matrix at a
time. The benchmark candidate uses batches of up to 16, reducing GEMM dispatch
without changing the optimizer equations. No production recipe enables batching
yet.

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
processes one matrix at a time. Shared offsets and batching are available only
as explicit benchmark experiments.

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
