# Simple LLM: eight-H100 and long-context training plan

This is a proposed plan for the modern **124,389,632-parameter** model, commonly
called our 100M model. It is not a 100B-parameter model. The earlier 500B-token
budget has been withdrawn; a replacement full-run token budget is not yet selected.

## Context progression

Start with a 2K–4K baseline, continue at **8K**, then evaluate a **16K** extension
through continued pretraining. Include genuinely long documents and code examples;
longer padded windows alone do not teach long-range use. Shorter examples remain
valid during later fine-tuning.

RoPE avoids a fixed learned positional-embedding table, but does not guarantee
quality beyond the context lengths seen during training. Evaluate any changes to
RoPE frequencies/scaling together with training at the intended lengths. Preserve
the original checkpoint configuration and record all context/RoPE changes.

## GPU scaling

1. Measure memory, actual nonpadding loss tokens/second, and validation quality
   on three H100 80GB GPUs at 4K, 8K and 16K.
2. Extend and qualify the launcher/configuration for **eight H100s on one host**.
   Verify the GPU interconnect; do not assume cross-host performance is equivalent.
3. Compare three and eight GPUs at matched context, data and optimizer token
   budgets before choosing the full-run configuration.

Use DistributedDataParallel, BF16 autocast, and an efficient SDPA backend; verify
which attention kernel actually executes. Reduce sequences per microbatch as
context grows, and add activation checkpointing if memory requires it. Flash
attention reduces attention memory traffic but does not remove quadratic attention
computation. Compare fixed target positions per update, and record the actual
nonpadding targets because document lengths and padding change the effective batch.

The existing checked-in launcher supports one through three GPUs. Eight-GPU
support, context-stage transitions, and the timings below are **not validated**
by this document. Exact resume currently rejects configuration changes; a context
extension needs an explicit, separately recorded continuation workflow rather than
silently editing a resume configuration. Parallel data preparation/prefetch must
also be qualified to keep the larger worker fed.

## Preliminary timing envelope

These are **unverified planning estimates**, not benchmark results or a launch
configuration. Assume eight H100s on one well-connected host, efficient attention,
and prepared data. Durations exclude corpus preparation and GPU queueing.

| Training exposures | 8K context | 16K context |
| --- | --- | --- |
| 5B tokens | 3–6 hours | 5–10 hours |
| 20B tokens | 12–24 hours | 20–40 hours |
| 50B tokens | 1.3–2.5 days | 2–4 days |

These ranges imply approximately 231K–463K loss tokens/second at 8K, and
139K–278K at 16K. They are assumed throughput envelopes, not extrapolations with
measured eight-GPU scaling factors. The 50B row is rounded from the same ranges.

The measured anchor is **345,895 nonpadding loss tokens/second on three H100s at
2,048 context** in the [500-update comparison](../results/scholarly-124m-gpu-comparison-20261017.md).
It does not establish throughput at 8K/16K or on eight GPUs. For provisional
scheduling only, allow about one day for 20B exposures at 8K, then replace this
estimate with measured long-context throughput before committing to the full run.

## Qualification before a full run

Measure sustained throughput after warmup, peak memory, actual SDPA backend, data
waits, and checkpoint/resume behavior. Compare held-out loss at matched lengths,
long-distance retrieval, and code completion requiring earlier context. An
accepted 16K input is not evidence that the model uses all 16K effectively.

Preserve the planned scholarly/code mixture and at most two passes per source.
A 20B-exposure run would require at least 10B eligible unique tokens overall and
sufficient capacity in each source share. The completed 5B experiment's roughly
2.75B unique-token corpus is insufficient for that target under the current policy.
