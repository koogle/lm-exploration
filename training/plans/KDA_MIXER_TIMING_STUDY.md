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
