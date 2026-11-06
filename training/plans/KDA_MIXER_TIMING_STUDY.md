# Mixer comparison and complete step-time accounting

November 6, 2026. Follow-up to the marginal consolidation results in
[KDA real-data MFU findings](KDA_REAL_DATA_MFU.md). Full pretraining stays held.
This work belongs to the stacked implementation PR #150; the planning/data
mixture PR #143 remains unchanged.

## Question and controlled comparison

The user's term “GDA” is provisionally interpreted as scalar-gated delta
attention (Gated DeltaNet); clarification against standard grouped-query
softmax attention is pending. The experimental scalar candidate retains the
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
