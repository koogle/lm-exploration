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
communication retain their existing precision. The next full-model comparison
uses tensorwise scaling and both scopes to test whether these kernel savings
survive in the real training step.
