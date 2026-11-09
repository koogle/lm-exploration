# Mixer comparison and complete step-time accounting

November 6, 2026. Follow-up to the marginal consolidation results in
[KDA real-data MFU findings](KDA_REAL_DATA_MFU.md). Full pretraining stays held.
This work belongs to the stacked implementation PR #150; the planning/data
mixture PR #143 remains unchanged.

## Question and controlled comparison

The user confirmed that “GDA” means scalar-gated delta attention (Gated
DeltaNet). The experimental scalar candidate retains the
18 recurrent / 6 full-attention layout, shared q/k/v projections, convolutions,
output gate, normalization, MLPs, vocabulary, MTP, packed documents and global
batch. One scalar decay per head replaces KDA's per-key-channel decay.

The scalar candidate uses FLA's gated-delta recurrence, a direct scalar gate
projection and AdamW for that projection. KDA uses its existing low-rank gate
projection and Muon. This is an architecture/implementation comparison, not an
algebraically equivalent kernel substitution. Training parameter counts are
959,843,116 for KDA and 953,069,572 for the scalar candidate. The MFU estimate
uses each model's actual matrix parameters, with the same H100 peak denominator.
Compare elapsed step time and tokens/s first; a change in counted FLOPs is not
itself a speedup. No checkpoint conversion or quality equivalence is claimed.

Use the frozen real 50/40/10 mixture, 4K context, eight strict 80GB H100s,
global rows 256 and microbatch six. Run KDA / scalar delta / KDA on the same
allocation, 20 steps each, excluding the first five. Preserve all losses,
gradient norms, memory, phase timings, hardware identity and source provenance.
Before that comparison, require scalar recurrence output/all-input gradient
checks against the expanded-channel reference, including document resets,
BF16 production gates and the initialized decay-rate range, plus a compiled
small-model check. CPU tests also cover document isolation and complete backward.
A faster mixer would require a separate matched learning study before adoption.

## Timing audit

The analyzer provides an exclusive device ledger for every recorded step:
compute-only, communication-only, memory-operation-only, overlap and idle.
Intervals are clipped to the step; concurrent streams count once in the ledger.
The sum must equal elapsed step time, with an explicit numerical residual.
Kernel-time flame widths separately sum kernel durations and therefore may
include overlap. CPU waits are context, not proof of removable device idle.

The first interactive view uses the existing optimized rank-zero trace. Across
three approximately 3-second windows it records 2,689–2,702 ms compute-only,
8.28–9.29 ms communication-only, 7.73–8.03 ms memory operations, 28.1–30.2 ms
overlap and 251–265 ms idle; residual is below 3e-10 ms. This closes accounting
for the recorded device, not causal attribution across all eight ranks.

A separate 16-step all-rank KDA diagnostic is launched as
`kda-allrank-profile-h100x8-20261106-01`, source `fd4de08`. It captures three
steady scheduled steps with CPU stacks/shapes, device streams and training
phase annotations on every rank. It has a 30-minute server execution timeout,
zero retries and an independent launch deadline. Profiled throughput is marked
invalid for qualification. Rank traces retain clock metadata; cross-rank timing
must be checked before inferring a shared critical path. Missing traces and
unattributed work must remain visible.

The viewer supports per-rank/per-step selection, zoom, search, launch/operation
attribution and flame views. Artifacts and raw traces belong on HF, with source
and profiling-scope provenance. No claim of a 40–60% MFU solution follows merely
from producing a flame graph: proposed changes must target measured costs and
then improve an unprofiled, repeated-control comparison.

## Implemented and running

The [first standalone interactive viewer](https://huggingface.co/datasets/jfrick/lm-exploration-results/resolve/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/rank0-timeline.html?download=true)
and its structured ledger/screenshot are archived under
[deep-dive](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive).
It adds detailed CPU/runtime lanes, inclusive CPU and GPU attribution flames,
recorded callsites and kernel-launch correlations. Host display truncation is
explicit; aggregate host totals retain all events. Idle-adjacent launches provide
context, not causal proof. The compressed standalone payload requires a modern
browser with native gzip decompression. Renderer validation covers all three
canvases and interactive selection without JavaScript errors.

Thirty-three focused CPU tests pass. Initial scalar H100 checks pass with maximum
normalized errors 0.00857 for recurrence/all-input gradients and 0.01377 for the
compiled model. The expanded production BF16/gate-range recurrence check passes
at 0.00756; its repeated compiled-model check also passes at 0.01377. These are
max-norm-normalized short checks, not per-element guarantees or learning results.

The matched throughput run is `kda-scalar-delta-h100x8-20261106-01`, source
`d9479aa`, with KDA/scalar/KDA arms and the same 30-minute worker timeout,
zero retries and independent launch deadline. Its completed speed result and corrected
all-rank audit are recorded below. Scalar gating is benchmark-only; the production
architecture and full-pretraining hold remain in effect.

## Completed matched mixer comparison

All three 20-step arms completed successfully on the same eight-H100 allocation,
with finite losses and gradient norms in all 60 steps. Medians exclude five
warmup steps per arm; control throughput drift is 0.224%.

| Arm | Tokens/s | Reported MFU | Median step | Peak allocated/GPU |
| --- | ---: | ---: | ---: | ---: |
| KDA first | 356,534 | 32.82% | 2.941 s | 65.50 GiB |
| Scalar-gated delta | 390,472 | 35.72% | 2.685 s | 63.41 GiB |
| KDA repeated | 355,736 | 32.68% | 2.948 s | 65.50 GiB |

Scalar gating improves throughput **9.52–9.76%**, cuts **256–262 ms/step**, and
saves **2.09 GiB/GPU**. Rank-zero phase medians locate most of the improvement:
forward/loss falls from 1.353–1.358 s to 1.288 s, local backward from
1.317–1.321 s to 1.148 s, and accumulated forward/backward from 2.821–2.827 s
to 2.569 s. Optimizer time falls from 104.8 to 100.5 ms. Phase medians need not
sum exactly to a median wall-clock step.

This is a meaningful speed result, not a convergence result. Final short-run
losses differ between architectures; the test does not qualify replacing KDA
for the main run. Even the faster candidate needs roughly another **287 ms**
removed per step to reach 40% under the current MFU estimate. No candidate has
been promoted and full pretraining remains held.

## Capture correction and exposed idle opportunities

The first all-rank attempt was canceled after identifying a scheduling bug:
`family.after_step` runs only on rank zero. A separate shared-loop diagnostic
`step_observer` now advances profilers on every rank, preserving the existing
rank-zero callback contract. A two-H100 integration check completed 12 steps
and produced all three expected windows (`ProfilerStep#6` through `#8`) on
both ranks. An initial small-check launch with insufficient source token budget
was canceled; prelaunch validation now rejects that mismatch. Failed/canceled
attempts remain archived.

The corrected eight-rank capture is `kda-allrank-profile-h100x8-20261106-02`,
source `f35951d`. It omits expensive CPU stack collection to reduce perturbation;
CPU operation/runtime ranges and launch correlations remain available. Detailed
profiling remains diagnostic, separate from unprofiled speed qualification.

The existing low-overhead rank-zero trace contains about 41,500 idle gaps per
step. Gaps below 5 microseconds sum to about 51 ms; 5–20 microseconds to 17 ms;
20–100 microseconds to 88–91 ms; 0.1–1 ms to 85–96 ms. Only four gaps per step
exceed 1 ms, totaling 8–10 ms. The largest are data fetching (4.5–5.9 ms) and
logging (1.6–1.8 ms), not a large input-loading stall.

Repeating device-adjacency patterns around KDA calls account for approximately
24–25 ms (`copy_` to KDA), 20 ms (`cat` to KDA), and 13–15 ms (`nonzero` to
compiled metadata work), each occurring 108 times per step. Host context shows
wrapper, allocation/reshape and compiler-cache work, with CUDA launch API calls
only several microseconds long. These support a narrowly scoped dispatch/setup
experiment with potential savings in tens of milliseconds. Adjacency is not
causality and profiling instruments these same wrappers. CPU synchronization
sums cannot be counted as removable wall time: 627 ms of `nonzero` host time
coincides with only 20 ms of GPU idle. Reaching 40% also requires reducing
busy-device work; CPU offload or loading changes do not explain the remaining gap.

## Completed eight-rank audit

Capture `-02` completed 16 steps with finite losses/norms and three complete
trace windows on **each of eight ranks**. Profiling remains marked unsuitable
for throughput qualification. Timed training steps 7–9 have median 3.031 s,
versus 2.940 s for post-capture steps 11–16: approximately **3.1% instrumentation
overhead**. Complete profiler windows include logging and average 3.037 s;
they deliberately differ from the trainer's `seconds` field. First-step startup
was 264.4 s and the second step 5.18 s before steady-state execution. This audit
covers training steps; evaluation/checkpoint overhead is outside this benchmark.

Mean exclusive time over the three windows, in milliseconds (rounded):

| Rank | Non-communication kernels alone | Communication alone | Memory operations alone | Cross-category overlap | Idle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 2621.0 | 42.1 | 7.8 | 124.5 | 241.9 |
| 1 | 2696.7 | 36.1 | 6.2 | 58.0 | 240.2 |
| 2 | 2605.8 | 70.1 | 6.0 | 124.3 | 231.0 |
| 3 | 2617.2 | 59.4 | 6.2 | 124.5 | 229.9 |
| 4 | 2614.7 | 60.0 | 6.0 | 125.1 | 231.5 |
| 5 | 2627.4 | 53.9 | 6.0 | 124.5 | 225.4 |
| 6 | 2690.2 | 34.0 | 6.3 | 67.8 | 239.1 |
| 7 | 2612.4 | 68.6 | 6.1 | 124.5 | 225.7 |

Every unrounded ledger sums to its complete window; residuals are numerical
roundoff. Non-communication kernels include memory-bound elementwise work,
not just tensor-core arithmetic. The GPUs show **92.0–92.6% activity**, while
the model FLOP estimate remains about 33% of BF16 peak. These measure different
things. Higher kernel-only time on ranks 1/6 mostly reflects less overlap, not
90 ms of slower arithmetic.

Independent kernel-duration sums are similar across ranks: GEMMs 1,384–1,408 ms,
compiled elementwise 413.5–414.5 ms, KDA 364.9–369.4 ms, KDA convolutions
90.5–91.3 ms and loss softmax-gradient 85.65–86.00 ms. These sums can overlap
and must not be added to the exclusive table. NCCL sums vary from 94 to 194 ms,
but only 34–70 ms occurs without other device work; communication time can also
include waiting. Large host sync durations likewise cannot be treated as savings.

Under the explicit same-Modal-worker/shared-host-clock assumption, observed
end-of-step device completion spans only 0.229–0.240 ms, with different latest
ranks across the three windows. There is no stable end-of-step straggler in this
capture. Synchronization can conceal earlier skew: this is not a calibrated
clock result or proof of a causal critical rank. The compact audit records the
assumption and validates rank/world/base-time metadata instead of inferring it
from the generic host label `modal`.

[All eight raw traces, ledgers and conditional alignment](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-allrank-profile-h100x8-20261106-02)
are archived with the [fresh representative rank-zero interactive viewer](https://huggingface.co/datasets/jfrick/lm-exploration-results/resolve/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-allrank-profile-h100x8-20261106-02/rank0-timeline.html?download=true).
The viewer is scoped to rank zero; the compact audit covers all eight. Kernel
and CPU attribution flames retain unattributed categories and distinguish
inclusive durations from exclusive elapsed time.

## What to fix next

The evidence supports reducing recurrent and elementwise execution cost, plus
small repeated dispatch gaps. It does not support input-loading work or CPU
offload as a route to the missing hundreds of milliseconds. Scalar gating has
already produced a measured 256–262 ms gain with the same context, batch and
sources, but needs a learning-quality comparison before a model decision.

For either architecture, a tightly scoped metadata/dispatch improvement can
be tested against the exposed tens-of-milliseconds budget above. If pursuing
the faster scalar candidate, its 2.09 GiB memory saving also motivates a bounded
larger-microbatch test; capacity and speed have not yet been established at that
batch. Larger remaining gains likely require reducing busy-device work, such as
a separately qualified precision or GEMM change. FP8 would change numerical
policy and the appropriate peak denominator, so evaluate actual tokens/s,
quality and per-precision accounting rather than claiming BF16 MFU improvement
by relabeling it. No proposed next experiment is automatically a production
change, and this evidence does not establish a route to 60% MFU.

## Detailed timing diagram and the remaining MFU gap

The [timing diagram](https://huggingface.co/datasets/jfrick/lm-exploration-results/resolve/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/timing-diagram/timing.png)
shows the exclusive ledger for all eight GPUs, the GPU-0 kernel groups, and the
largest individual kernels outside GEMMs and NCCL. The same HF folder contains
an SVG and the complete kernel names, durations and call counts as JSON.
The existing interactive timeline remains the chronological view; the stacked
bars in this diagram regroup time by activity and are not chronological.

Rebuild from the archived capture with matplotlib installed:

```sh
python -m model_hybrid.profile_plot \
  --trace /path/to/trace.json.gz \
  --headline /path/to/all-rank-headline.json \
  --output /path/to/timing
```

The name classifier previously placed `intra_backward` (112.65 ms) and
`chunk_gla_fwd_kernel_o` (20.07 ms) in `other`. Both belong to the current KDA
implementation. The corrected offline diagram places them under recurrence;
future benchmark summaries use the same classifier. Original archived summaries
retain their historical classification. Clipping kernels to the exact windows
also introduces tiny boundary differences from the original profiler summary.

| GPU-0 kernel group | ms/step, summed kernel durations |
| --- | ---: |
| Dense matrix kernels, including heads and optimizer matrix work | 1,394.1 |
| KDA recurrence, gates and L2 normalization | 500.1 |
| Compiled elementwise operations, reductions and casts | 414.1 |
| Other native kernels | 168.0 |
| NCCL collectives, including possible waiting | 166.5 |
| KDA convolutions | 91.1 |
| Loss softmax gradient | 85.7 |
| Full attention and fused surrounding operations | 84.0 |
| Copies, fills and memory operations | 9.9 |
| Explicit optimizer/multi-tensor kernels | 6.4 |

These rows overlap across streams. They must not be added to the exclusive idle
ledger or interpreted as removable time. In particular, the full optimizer is
about 105 ms by phase timing; its matrix and elementwise operations appear in
other rows. The KDA row excludes projection GEMMs. The largest native kernel in
`other` is FP32 elementwise addition (87.7 ms); its name alone does not identify
which functional use of addition is responsible.

The measured GPU activity is approximately 92%, but activity means that at least
one recorded kernel or memory operation is active. It does not establish tensor
core saturation, achieved memory bandwidth, occupancy, or useful model FLOPs.
Only about 1.39 seconds of the approximately 3.04-second profiled window is dense
matrix-kernel duration. Recurrence and elementwise operations are necessary work,
but much of their arithmetic is absent from our conventional MFU numerator.
Neither “all non-GEMM time is wasted” nor “92% activity means efficient training”
is supported by this trace.

The scalar Gated DeltaNet comparison is related to MFU through the formula:

`MFU = tokens/sec × estimated model FLOPs/token / aggregate GPU peak FLOPs/sec`.

The matched first KDA control achieved 356,534 tokens/sec and 32.82% MFU; scalar
delta achieved 390,472 tokens/sec and 35.72%. That is **9.52% more throughput**,
and **2.90 percentage points / 8.85% relative improvement in MFU**. The ratios
are slightly different because scalar delta has a smaller estimated FLOPs/token
numerator and the reported metrics are independently summarized medians.
This is an architecture-and-implementation comparison, not an equivalent-model
kernel optimization. It provides no long-run quality equivalence evidence.

At fixed KDA work per step, 40% MFU requires reducing the 2.941-second unprofiled
control to approximately 2.413 seconds: **528 ms saved, or 18% lower latency**.
60% requires approximately 1.609 seconds: **1,332 ms saved, or 45% lower latency**.
The approximately 242 ms idle in the instrumented GPU-0 window is far smaller
than either gap. Even perfect removal of idle time cannot deliver 40% on its
own. Scalar delta still needs approximately 287 ms saved to reach 40% under its
own estimator. These are fixed-work arithmetic targets, not measured forecasts.

The largest measured targets are therefore recurrence (especially the 113 ms
intra backward and 101 ms WY backward kernels), elementwise/reduction/cast work,
and the dense output/body/optimizer matrix work. Fusion, less materialization,
and better tiling are hypotheses to evaluate against those costs. CPU offload
would add transfers and dependencies for activations needed by the next GPU
operation; this trace supplies no evidence that it would help. Data-fetch and
scatter phase timings are only approximately 8 ms combined.

The trace establishes where time runs, but does not yet prove why each large
kernel is slow. A defensible next kernel change needs achieved DRAM bandwidth,
tensor-core activity, register/shared-memory limits and stall evidence for the
specific kernel, together with a same-model unprofiled end-to-end comparison.
No claim that 40–60% is attainable with the present architecture follows from
these timing measurements alone. This diagram uses the existing captures and
requires no new GPU run.

## BF16 versus FP8 linears on eight H100s

The Inkling 2 Small review and lower-precision rationale are in
[reference PR #151](https://github.com/koogle/lm-exploration/pull/151). Its reported
native NVFP4/GB300 training motivates investigation, but the detailed recipe is
not yet disclosed and H100 lacks Blackwell's native FP4 matrix hardware.

Our run `kda-fp8-linears-h100x8-20261107-02` (source `e1c9a4b`) compared
BF16 / torchao 0.18.0 rowwise FP8 / BF16 in one strict eight-H100 allocation.
The 960M KDA, real 50/40/10 data, seed, context 4096, global batch 256, microbatch
six, optimizer and full MTP loss stayed fixed. Only 183 eligible large linear
layers changed computation precision; recurrent state, functional vocabulary
heads, optimizer and gradient communication retained their existing precision.
Both controls use the same dependency image as FP8. Parameter identities and
counts were checked; FP8 did not reduce the model size.

[Evidence and provenance](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-fp8-linears-h100x8-20261107-02)
include every arm's log, numerical checks and an earlier launch error. Attempt 01
failed before GPU-worker submission because Modal's `with_options` does not
accept an image override. The corrected image pins torchao before registration.
The successful worker retained its 1,800-second timeout, zero retries and an
independent overall deadline supervisor.

| Arm | Median step ms | Tokens/sec | BF16-equivalent MFU | Peak allocation/GPU |
| --- | ---: | ---: | ---: | ---: |
| BF16 first | 2,918.0 | 359,347 | 33.05% | 65.50 GiB |
| FP8 rowwise linears | 3,612.1 | 290,298 | 26.79% | 59.88 GiB |
| BF16 repeat | 2,917.5 | 359,408 | 33.12% | 65.50 GiB |

Each arm completed 30 steps, with five excluded from timing; all 90 steps had
finite losses and gradient norms. The FP8 linears' representative output/dX/dW
relative-L2 errors were 3.57–3.78%. Last main losses were 11.33825, 11.35398 and
11.33803. The short run is still within LR warmup and does not establish quality
equivalence or stability over the proposed pretraining budget.

FP8 throughput regressed 19.22–19.23%; control drift was only 0.017%. Memory fell
5.62 GiB/GPU. Local-backward phase median grew 1,301 → 1,900 ms, synchronizing
backward 153 → 202 ms, forward/loss 1,346 → 1,391 ms, and optimizer stayed ~105 ms.
These phase medians are non-additive; synchronizing backward includes computation.
The run measures the regression, not a kernel-level causal explanation.

Keep BF16 as the qualified default. The FP8 implementation remains benchmark-only.
A subsequent investigation can isolate backward quantization/layout and GEMM costs,
or test whether the saved memory enables a sufficiently faster larger microbatch.
Neither benefit is established here. Do not reinterpret the reported MFU as native
FP8 hardware utilization: the existing estimator retains its BF16 peak denominator.

Reproduce the bounded comparison from `training/`:

```sh
modal run -m model_hybrid.modal_tools::bench \
  --config hybrid/configs/kda-960m-100b.json --gpus 8 --steps 30 \
  --microbatch 6 --global-rows 256 --real-data --require-gpu-model \
  --phase-timings --compare-fp8 --run-id YOUR_UNIQUE_RUN_ID
```

## FP8 backward diagnosis: scaling consumes the GEMM savings

The one-H100 studies (launch sources `51009d6` and `3d8aca7`) compare compiled
linears with identical FP32 weights, BF16 inputs and upstream gradients. Main
microbatch shapes use 24,576 token rows; tail shapes use 8,192. Four warmups precede
20 unprofiled CUDA-event samples. A separate single-backward profile attributes
kernel duration; profiler times are not substituted for unprofiled latency.
[Complete results and logs](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/fp8-backward-study)
retain both studies, including the unsuccessful alternatives.

For the second study's `[24576, 1536] @ [1536, 4096]` MLP expansion:

| Recipe | Backward median ms | Profile: GEMM ms | Profile: compiled scaling/conversion ms |
| --- | ---: | ---: | ---: |
| BF16 | 0.791 | 0.768 | 0.011 |
| Original rowwise FP8 | 1.114 | 0.421 | 0.676 |
| Tensorwise FP8 | 0.695 | 0.416 | 0.261 |
| Rowwise FP8, higher-precision dW | 0.856 | 0.587 | 0.267 |
| Rowwise FP8 without power-of-two scale rounding | 1.018 | 0.426 | 0.577 |

The FP8 GEMMs are already substantially faster than BF16. Rowwise scaling,
conversion and transposition kernels erase that gain. Removing power-of-two
rounding recovers some time but leaves most of the overhead. Retaining BF16
weight-gradient computation improves over original rowwise FP8 but is slower
than tensorwise on the dominant MLP shapes. Kernel names and durations in the
archive expose the fused `amax`, scale, cast and transpose operations; this is
measured device time, not a claim of a particular achieved DRAM bandwidth or
hardware stall reason.

Tensorwise reduces expansion backward by ~38% versus rowwise. For the MLP
contraction `[24576, 4096] @ [4096, 1536]`, backward falls **1.131 → 0.573 ms**
(~49%). Smaller projections and tail microbatches do not uniformly improve over
BF16. Standalone calls also expose different host-dispatch overhead than a whole
compiled model, so summing these medians cannot predict a training-step speedup.

All 40 cases in the second study passed the finite/relative-L2 smoke gate.
Tensorwise uses E5M2 gradient operands and has ~5.9% dX/dW relative-L2 error on
these Gaussian inputs, versus ~3.8% for rowwise E4M3. Its forward error is ~3.7%.
This is an intentional numerical-policy change, not an algebraically identical
kernel rewrite. It needs learning-curve validation before production adoption.

The benchmark now accepts `--fp8-recipe` and `--fp8-scope`. Scope `all` converts
183 large linears; `mlp` converts 75 MLP linears (471,859,200 weight elements),
retaining BF16 projections. `both` on the Modal entrypoint compares both scopes
between two BF16 controls. The recurrence, vocabulary losses, optimizer and
communication retain their existing precision. The full-model comparison below
uses tensorwise scaling and both scopes to test whether these kernel savings
survive in the real training step.

### Eight-H100 tensorwise comparison and the median trap

Run `kda-fp8-tensorwise-h100x8-20261107-01`, source `eeac8de`, completed 20
steps per arm on one strict eight-H100 allocation. All 80 steps had finite main
and MTP losses and gradient norms. The configuration, real data, seed and batch
sizes match the earlier FP8 comparison. Five steps per arm are excluded below.
[Raw logs and aggregate audit](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-fp8-tensorwise-h100x8-20261107-01)
retain every measured step, including stalls.

| Arm | Median step ms | Aggregate tokens/sec | Aggregate BF16-equivalent MFU | Peak allocation/GPU |
| --- | ---: | ---: | ---: | ---: |
| BF16 first | 2,921.3 | 359,048 | 33.04% | 65.50 GiB |
| Tensorwise FP8, all eligible linears | 2,777.6 | 344,153 | 31.67% | 59.88 GiB |
| Tensorwise FP8, MLP only | 2,816.0 | 372,160 | 34.25% | 62.18 GiB |
| BF16 repeat | 2,919.0 | 359,223 | 33.06% | 65.50 GiB |

MLP-only FP8 improves aggregate throughput **3.60–3.65%** and saves **3.32 GiB/GPU**.
Its local-backward median is **1,220 ms**, versus **1,315–1,317 ms** for BF16.
Last main losses are 11.97560 / 11.97971 / 11.97671 / 11.97569 in table order.
These remain short, early-warmup training comparisons, not held-out quality tests.

All-linear tensorwise FP8 has a genuinely faster local-backward median (**1,176
ms**) and normal steps around 2.77 seconds, but five of its 15 measured steps
stall in forward. The delayed rank's forward takes approximately 0.8 seconds
longer; other ranks wait in synchronizing backward. Every rank is delayed once
across steps 6, 7, 10, 11 and 12. The faster median therefore does **not** imply
higher aggregate throughput over this window. Do not omit these steps after
seeing the data.

The shared benchmark now reports timed tokens, timed seconds, aggregate
throughput and time-weighted MFU alongside its median metrics. A warmup-only
run has unavailable timing metrics and is not throughput-valid. A regression
test ensures that stalls lower the aggregate result even when the median stays
fast. This fixes reporting; it does not change the underlying training speed.

### Host-pause diagnosis

A separate one-H100 run, `kda-fp8-host-h100-20261107-01` (source `b5d7e39`),
uses 32 global rows / microbatch six to match one rank's work and records Python
GC and Dynamo compilation callbacks. It completes 40 steps. At step 30,
**generation-2 GC takes 739.6 ms**, matching the increase from approximately
2.80 to 3.55 seconds. No post-warmup Dynamo compile callback is recorded. This
identifies that run's stall; these hooks do not inventory every backend compiler
and are not a direct trace of the earlier eight-H100 pauses.
[The diagnostic evidence](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-fp8-host-h100-20261107-01)
includes the callback events and per-step timings.

The benchmark-only candidate `--freeze-gc-after-warmup` collects once after step
five, freezes the remaining live objects out of future cyclic-GC scans, and
barriers all ranks before step six. Automatic GC stays enabled for newly
allocated objects. The policy records its startup collection cost and unfreezes
objects when the benchmark finishes. It is applied to BF16 and FP8 alike in a
fresh comparison; this is a collect-plus-freeze policy, not an isolated proof
that freezing alone is necessary. Its cost is outside the steady timing window,
not claimed to disappear. Long-run host-memory behavior and checkpoint/evaluation
lifecycle behavior still need qualification before use in production training.

### Matched eight-H100 comparison with explicit warmup cleanup

Run `kda-fp8-gc-h100x8-20261107-01`, source `0d3ab12`, applies that same cleanup
policy to BF16 / tensorwise FP8 (all eligible linears) / BF16 on one strict
eight-H100 allocation. Each arm completes 40 steps; the first five are excluded
by the predetermined timing window. All 120 steps have finite main/MTP losses
and gradient norms. The real 50/40/10 corpus, seed, model, global batch 256,
microbatch six and 4K context are unchanged.
[Raw results, numerical checks, per-rank host events and timing audit](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-fp8-gc-h100x8-20261107-01).

| Arm | Aggregate tokens/sec | Aggregate BF16-equivalent MFU | Median / maximum timed step | Peak allocation/GPU |
| --- | ---: | ---: | ---: | ---: |
| BF16 first | 358,215 | 33.00% | 2.928 / 2.944 s | 65.50 GiB |
| Tensorwise FP8, all eligible linears | 377,293 | 34.76% | 2.778 / 2.806 s | 59.88 GiB |
| BF16 repeat | 358,170 | 33.00% | 2.927 / 2.951 s | 65.50 GiB |

FP8 improves aggregate throughput **5.33–5.34%** over both controls and saves
**5.62 GiB/GPU**. Control drift is **0.012%**. All 35 timed steps per arm count
toward aggregate throughput, with no post-hoc outlier removal. No timed GC event
of at least 10 ms or Dynamo compile callback is recorded on any rank. These
hooks do not rule out every possible host pause or backend compilation event.

| Rank-zero phase median | BF16 first | FP8 | BF16 repeat |
| --- | ---: | ---: | ---: |
| Forward and loss | 1,355 ms | 1,353 ms | 1,358 ms |
| Local backward | 1,317 ms | 1,180 ms | 1,318 ms |
| Synchronizing backward | 128 ms | 122 ms | 127 ms |
| Optimizer | 105 ms | 105 ms | 105 ms |

Local backward is **10.4–10.5% faster**, saving approximately **137–139 ms**.
The unchanged forward/loss and optimizer work limits the full-step benefit.
These are independent phase medians, not an additive exclusive ledger;
synchronizing backward includes computation, communication and waiting.

The explicit post-step-five collection/freeze costs **0.750–0.800 s/rank** for
the first BF16 arm, **0.885–1.136 s/rank** for FP8 and **0.603–0.693 s/rank** for
the repeated BF16 arm, followed by an all-rank barrier. This one-time startup
cost is outside the measured steady window and is not eliminated work. The
earlier short-run stall penalty should not be extrapolated as a recurring
long-run tax; this experiment qualifies the combined cleanup policy over 35
timed steps, not the necessity of freezing over a collect-only alternative.

Last main losses are **10.65804 / 10.66881 / 10.65812**. The LR is still in
early warmup. Tensorwise E5M2 gradient conversion had approximately 5.9%
relative L2 errors on the isolated Gaussian checks, versus approximately
3.7% for rowwise E4M3. Finite short runs do not establish learning-quality
equivalence, robustness to trained activation outliers, or long-run host-memory
behavior. FP8 and the GC policy remain benchmark-only; production BF16 and
KDA defaults are unchanged. All comparison apps are stopped.

This is a measured FP8 backward improvement, but **34.76% BF16-equivalent MFU
is still below 40–60%**. It uses the existing 989 TF/s/GPU BF16 peak denominator,
not native FP8 peak utilization. Reaching 40% at the same estimated work still
requires roughly another 0.36 seconds removed from each 2.78-second step.

### Forward/loss kernel experiment: vector-grouped reduction

The vocabulary loss performs its hidden and weight gradients inside its forward
call. Its contribution to `forward_and_loss` is therefore more than inference
forward work. A candidate groups the vector-load elements before merging their
online softmax statistics, reducing serial per-element updates without changing
the loss objective or logits traffic. Source `ac702d9` retains this experiment;
it was discarded from active code after measurement.

On H100 at the production BF16 shape `[2048, 154880]`, reference / grouped /
reference-repeat median kernel times are **0.67136 / 0.66909 / 0.67133 ms**.
The approximately **0.34%** difference is not a useful optimization. At input
scale 20 the difference is similarly small. BF16/FP32, aligned production
vocabulary and an unaligned 8,193-token vocabulary, ignored labels, and two
input scales all pass finite/numerical checks. Maximum production BF16 gradient
relative L2 difference is below 9e-8. These are isolated kernel measurements;
there is no claimed full-model throughput gain.

The production BF16 kernel's logical two reads plus one write move about 1.90 GB
in 0.671 ms, approximately **2.84 TB/s**. This is logical traffic, not a DRAM
counter measurement, and explains why changing the reduction arithmetic gives
little benefit. The previously measured ~86 ms/step kernel total is itself
under 3% of step time. No full-model comparison was launched for this negligible
candidate. [Checks, timings and provenance](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/grouped-softmax-forward-20261107)
preserve the negative result.

### Shared-loop GC fix and explicit FP8 MFU measurement

Run `kda-fp8-mfu-shared-gc-h100x8-20261107-01` (source `665627e`) completes
100 real-data FP8 steps on eight exact H100s, with 95 predetermined timed steps.
The collect/freeze policy now lives in `trainer.loop`, rather than the benchmark
observer. Hybrid configs use `gc_warmup_steps=5`; zero disables it. The session
warmup restarts after resume, automatic collection remains enabled for new
objects, and the loop unfreezes the permanent generation during `finally`
cleanup. Current resident CPU memory is logged on every rank.

| Measurement | Result |
| --- | ---: |
| Aggregate throughput, including every timed pause | 371,739 tokens/sec |
| BF16-equivalent MFU, 989 TF/s/GPU denominator | 34.27% |
| All-FP8 peak denominator, 1,978 TF/s/GPU | 17.14% |
| Mixed-precision ideal-time estimate | 24.31% |
| Estimated model arithmetic per GPU | 338.98 TFLOP/s |
| Estimated FP8 share of model FLOPs | 58.15% |
| Median / maximum timed step | 2.810 / 3.635 seconds |
| Peak allocated GPU memory | 59.88 GiB |

The mixed estimate weights FP8 FLOPs by 1,978 TF/s and the remaining modeled
BF16 FLOPs by 989 TF/s. This avoids treating the BF16 vocabulary heads as FP8.
The code reports all three denominators explicitly; a denominator change is not
a speed improvement. These are model-FLOP estimates, not hardware-counter tensor
utilization. They retain the existing omissions of recurrence, conversions,
elementwise and optimizer FLOPs and the sampled attention-length estimate.

All 100 main/MTP losses and gradient norms are finite; final main loss is
7.25422. No timed GC event of at least 10 ms or Dynamo compile callback is
recorded on any of the eight ranks. The one slow timed step, **step 90**, spends
**840 ms in data fetch**; its forward/backward phases remain normal. It remains
in the throughput/MFU totals. This is not evidence that every type of pause is
eliminated.

Host RSS at step six is 8.30 GiB on rank zero and 7.13–7.17 GiB on other ranks.
No rank exceeds its step-six RSS during the remaining measured window; end RSS
is flat or slightly lower. Startup collection costs **0.836–0.958 seconds/rank**,
plus a barrier, and the first compilation/startup step takes **390.4 seconds**.
Those costs are outside steady-state MFU, not claimed as saved work. This short
window does not establish multi-day memory stability or long-run FP8 learning
quality. There is no same-allocation BF16 control in this follow-up; use the
earlier matched comparison for the speedup claim.

Seventeen focused CPU tests pass, including precision partitions on the real
960M model, the high-precision weight-gradient recipe, new-cycle collection and
release of the frozen graph at session exit. Ruff passes. [Full results and
per-rank audit](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-fp8-mfu-shared-gc-h100x8-20261107-01)
retain startup costs, the fetch pause and source provenance.

### Fresh FP8 forward/loss attribution

Run `kda-fp8-forward-profile-h100-20261107-01` (source `9388142`) captures
three instrumented steps within a 12-step run on one H100. Global rows 32 and
microbatch six reproduce one rank's shapes in the eight-H100 run. CPU launch
correlations attribute the following GPU kernel sums to `forward_and_loss`:

| Kernel category | Summed ms per profiled step |
| --- | ---: |
| Matrix multiplies | 654.1 |
| Compiled elementwise/conversion | 265.5 |
| KDA recurrence | 143.6 |
| Loss softmax/gradient | 85.5 |
| KDA convolution | 27.2 |
| Full attention | 20.7 |
| Other and memory operations | 9.4 |

These are kernel sums attributed by launch, not an exclusive wall-time ledger;
the single-GPU profile does not measure eight-GPU communication or replace the
uninstrumented MFU result. The vocabulary-head GEMM kernel families contribute
roughly **501 ms** of this forward/loss attribution (including loss-gradient
work). The largest individual compiled FP8 cast/layout kernel accounts for
38.7 ms; one amax-reduction kernel accounts for 24.0 ms. These are concrete
remaining targets, not claimed optimizations.

The vocabulary-head GEMMs are a larger remaining opportunity than further
softmax arithmetic tuning. They still use BF16; changing their precision would
require separate loss/gradient and learning-quality qualification. This study
establishes no additional forward-kernel speedup. [Raw timeline, per-kernel
attribution and provenance](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-fp8-forward-profile-h100-20261107-01)
are archived. All three follow-up apps are stopped with zero tasks.

### Mixed-precision vocabulary kernels

The next experiment changes the vocabulary GEMMs themselves. Custom Triton
kernels compute partial amax, a scale, and a fused scale/clamp/FP8 cast, avoiding
full-size FP32 conversion intermediates. The production chunked loss loop is
shared. E4M3 operands feed the logit GEMM; the optional hidden-gradient GEMM uses
E5M2 logit gradients and E4M3 weights. FP32 softmax statistics and FP32
weight-gradient accumulation from BF16 operands are retained. Converted weights
and layouts are reused within one loss invocation, never across optimizer steps.

Two isolated H100 studies measure all loss and gradient work, including these
conversions. At 24,576 tokens, width 1,536 and vocabulary 154,880, the second
study's normal-input BF16 controls take **61.58 / 61.98 ms**, FP8 logits only
**54.95 ms**, and FP8 logits plus hidden gradients **53.56 ms**. That is about
**13–14% less complete head time** for the latter, not a full-model speedup.
The first study measures 63.56 / 64.06 ms versus 54.13 ms, a similar direction.
The smaller 8,192-token tail benefits less because conversion/layout overhead
is less amortized.

Normal-input gradient errors for the aggressive variant are approximately
2.9% relative L2 for hidden gradients and 0.03% for weight gradients. Peaked
inputs increase weight-gradient error to about 7.9%. A severe synthetic outlier
case in the first study reaches **66% weight-gradient error** despite only a
small relative change in average loss. These are stress inputs, not captured
production activations, but they prevent claiming numerical or learning-quality
equivalence. Results vary across random inputs; the less severe second outlier
result does not supersede the failure.

The conservative `dx` variant preserves BF16 logits, loss and weight gradients
exactly for fixed inputs/weights in these tests. Hidden-gradient errors range
from 2.9% to 5.0%. It saves only about 1–2% on the large-head operation and is
slower on the tail, so it is not the candidate for the full performance test.

Compiled forward/backward smoke checks pass for all three variants, including
zero hidden gradients at ignored positions. An initial compile check failed
because a custom-op argument named `mode` collided with PyTorch's internal
functionalization dispatch; renaming it to `precision` fixes the failure.
The failed log and source remain archived with the successful checks.
[Isolated results and provenance](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/fp8-vocabulary-kernels-20261108).

All head variants remain opt-in benchmark experiments. The full-model comparison
uses the aggressive variant to measure the available speed benefit, not to
approve it for the long pretraining run.

The first eight-H100 attempt, `kda-fp8-head-h100x8-20261108-01`, completes its
40-step BF16-head control at 373,966 tokens/sec but stalls during the candidate's
first step. A live stack capture shows rank four waiting for an Inductor compile
future. Its forked compiler child is blocked in
`ProcessPoolExecutor`'s `weakref_cb`, acquiring `shutdown_lock` during
`ast.parse`; another rank is waiting in backward. The compiler pool has 32
workers per rank. This is evidence of a compiler-process/GC deadlock, not a
kernel throughput result. The attempt is canceled before its server timeout.
[Partial control, launch log and captured stacks](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-fp8-head-h100x8-20261108-01)
are preserved. Source `0398c89` changes the shared loop to fresh spawned
compiler interpreters and two workers per rank; both controls and the candidate
use that same startup policy in the retry.

The retry, `kda-fp8-head-h100x8-20261108-02`, completes all three 30-step arms on
eight H100s. Modal preempts the first container during startup and automatically
restarts it; the launch log preserves that interruption. All completed arms use
the replacement allocation, real 50/40/10 data, context 4096, global rows 256,
microbatch six and the same tensorwise-FP8 trunk. Only the vocabulary-head
precision changes. Each aggregate includes all 25 predetermined post-warmup
steps, including any stalls.

| Head precision | Aggregate tokens/sec | BF16-equivalent MFU | Median step | Peak allocated/GPU |
| --- | ---: | ---: | ---: | ---: |
| BF16 first | 374,058 | 34.45% | 2.801 s | 59.88 GiB |
| FP8 logits and hidden gradients | 385,579 | 35.51% | 2.718 s | 60.32 GiB |
| BF16 repeat | 373,242 | 34.37% | 2.809 s | 59.88 GiB |

The head change improves full-model throughput **3.08–3.31%**, with 0.22% control
drift. Rank-zero forward/loss medians fall from 1366 / 1361 ms to 1310 ms;
local backward medians are 1186 / 1185 ms versus 1153 ms. These phase observations
are not a new per-kernel causal profile, and independent medians do not add to a
complete step. Optimizer medians remain about 105 ms. Additional quantized weight
layouts increase peak allocation by **0.44 GiB/GPU**.

All 90 steps have finite loss, MTP loss and gradient norm. All eight ranks report
host diagnostics for each arm, with no timed GC event at the diagnostic threshold
or compile callback. First-step startup costs are 367.45 / 147.22 / 52.87 seconds,
respectively; cache reuse makes these unsuitable as matched startup comparisons.
The spawn-worker change completes this previously failing comparison, without
establishing that every possible compiler hang is eliminated. The app is stopped.

FP8's share of modeled FLOPs rises from 58.22% to 84.35%. Consequently,
precision-weighted ideal-time MFU falls from 24.42% to 20.53%, despite higher
throughput and BF16-equivalent MFU. The candidate is 17.75% against the dense FP8
peak. These denominators describe different ceilings; none is a hardware-counter
measurement. The result still falls short of 40–60% on the fixed BF16 basis.

Final losses are 11.34875 / 11.35714 / 11.34962. Thirty early-warmup steps do not
establish learning-quality parity, and they do not supersede the synthetic
outlier-gradient failure above. Keep this head precision experimental. A useful
next numerical investigation is scaling granularity on captured training
activations; simply enabling the faster path is not justified by this test.

Ten focused CPU tests and Ruff pass, alongside the archived compiled GPU checks.
[Complete matched results, host audit and provenance](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-fp8-head-h100x8-20261108-02).

## Shared document metadata and microbatch study (November 9)

The metadata experiment lazily computes packed-document offsets once in each
forward's `DocumentMask` and supplies that tensor to every recurrent layer.
The optional offsets argument also serves the scalar-delta benchmark. A compiled
H100 correctness check compares two different packed layouts, MTP losses and all
parameter gradients with per-layer construction: all values are finite and the
largest absolute difference is 1.91e-6. Eighteen focused CPU tests and Ruff pass.
[Correctness check](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-metadata-check-20261109).

The first three-arm run appears to improve throughput from 341,451 / 341,659 to
351,084 tokens/sec. However, its reference rebuilds metadata at the caller rather
than inside the original recurrence helper. That changes the compiler boundary;
this is not a valid speedup over the original implementation. Preserve this
result as a control-design failure, not a qualified optimization.
[Initial comparison](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-metadata-h100x8-20261109-01).

Source `21ee6b8` restores the reference calculation boundary. The follow-up uses
the same eight physical H100s, FP8 tensorwise trunk, BF16 head, real 50/40/10
data, context 4096, global rows 256 and microbatch six. Each completed arm has
25 steps, with all 20 predetermined post-warmup steps included:

| Arm | Aggregate tokens/sec | BF16-equivalent MFU | Peak allocated/GPU |
| --- | ---: | ---: | ---: |
| Corrected original reference | 376,437 | 34.64% | 59.88 GiB |
| Shared metadata | 351,481 | 32.35% | 67.47 GiB |

Shared metadata is **6.63% slower**, with about 7.59 GiB more peak allocation.
Rank-zero forward/loss medians are nearly unchanged (1349 versus 1347 ms), while
local backward rises from 1179 to 1362 ms. No timed host GC or compile event is
recorded; all 50 completed steps are finite. These observations suggest a
compiled-graph/saved-activation interaction, but do not identify its exact cause.
The remaining repeat/larger-batch arms are canceled before timeout after this
regression; the archive explicitly marks the comparison incomplete. This is
evidence to reject the change, not a completed three-control qualification.
[Corrected partial comparison and launch log](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-metadata-batch-h100x8-20261109-02).

Source `2caf164` retains shared metadata only behind the benchmark's explicit
`--shared-kda-metadata` flag. Normal training and direct benchmarks use per-layer
construction. The subsequent microbatch comparison uses that restored path.
The launcher carries the same FP8 settings into the larger-batch and repeated
control arms, avoiding an accidental precision change. Global batch and data
stay fixed; dynamic FP8 scaling can still change rounding with microbatch size.

The completed `kda-batch-h100x8-20261109-03` comparison uses the restored original
path on the same eight H100s. All arms retain the FP8 tensorwise trunk and BF16
head. Global rows remain 256: each rank's accumulation changes from
`[6, 6, 6, 6, 6, 2]` to `[8, 8, 8, 8]`.

| Microbatch | Aggregate tokens/sec | BF16-equivalent MFU | Median step | Peak allocated/GPU |
| --- | ---: | ---: | ---: | ---: |
| Six, first | 376,114 | 34.61% | 2.787 s | 59.88 GiB |
| Eight | 395,245 | 36.37% | 2.653 s | 73.43 GiB |
| Six, repeat | 376,090 | 34.61% | 2.789 s | 59.88 GiB |

Eight improves aggregate throughput **5.09%** over both controls; control drift
is 0.0064%. All 75 steps have finite loss, MTP loss and gradient norm, and every
arm includes all 20 predetermined timed steps. All eight ranks report host
diagnostics with no timed GC event at the diagnostic threshold or compile
callback. Final losses are 11.77822 / 11.77732 / 11.77715; this short test is not
long-run FP8 quality qualification.

Rank-zero forward/loss medians decrease from 1347 / 1348 ms to 1276 ms. Local
backward decreases from about 1180 to 911 ms, while the synchronized backward
phase increases from 136 / 137 to 347 ms: the final microbatch now contains eight
rows rather than two. The local-backward decrease alone therefore overstates
the gain. Accumulation phase medians decrease from 2666 / 2668 to 2533 ms, and
optimizer time stays near 105 ms. Independent phase medians are not additive.

Use microbatch eight for subsequent benchmarks with this FP8-trunk/BF16-head
configuration and the original per-layer metadata path. It fits this bounded
test with 13.55 GiB more peak allocation; this does not qualify eight for the
all-BF16 production configuration or prove long-run memory stability. The
36.37% BF16-equivalent MFU remains below 40%; precision-weighted ideal-time MFU
is 25.78% and dense-FP8-peak MFU is 18.19%. All test apps are stopped with zero
tasks. Production data, architecture and precision settings remain unchanged.

[Complete microbatch comparison, host audit and provenance](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-batch-h100x8-20261109-03).
