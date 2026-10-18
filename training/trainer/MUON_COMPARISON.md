# Fresh 124.4M Muon versus AdamW comparison

This is a stacked follow-up to PR #17 (`training/padding-optimizations`). It uses
Simple LLM's existing scholarly/code raw snapshot, frozen 32K tokenizer, 14×768
architecture, 8K context and global batch of 32 windows. Both arms start with seed
42 and a fresh data cursor. Neither inherits weights from the previous AdamW run.

The initial bounded comparison is **100M nonpadding tokens per optimizer**, not a
5B-token production run. Both use eight H100s, microbatch 2, trailing-padding
cropping, cached RoPE, `torch.compile(dynamic=True)`, prefetch depth 4 and no chunked
loss. Compiler threads are limited to one per rank. A complete batch that would
exceed the ceiling is not consumed; both arms must finish at identical exposure.

Muon updates only hidden block matrices. The tied embedding/output table, norms
and biases use fused AdamW (betas 0.9/0.95). The baseline uses fused AdamW for all
parameters. Muon uses Nesterov momentum 0.95, five BF16 quintic Newton–Schulz
iterations and original aspect-ratio scaling `sqrt(max(1, rows/columns))`, following
[the Muon reference](https://github.com/KellerJordan/Muon). It is replicated on
all DDP ranks after gradient reduction, with no optimizer sharding.

Muon peak LR is 0.02; the AdamW peak LR is 0.0003 in both arms. Both schedules use
the same 20M-token warmup and cosine decay to 10% over 100M tokens. Weight decay
is 0.1 and global gradient clipping is 1.0. Different learning rates imply different
absolute decoupled decay strengths. These are starting hyperparameters, not tuned
optimizer optima. The experiment compares these recipes, not every possible Muon
and AdamW configuration. Repeated seeds and larger validation are future work.

From `training/`:

```sh
uv sync --locked --extra corpus --extra research
uv run modal run modal_muon_comparison.py --prefix muon-124m-100mt-<fresh-id>
```

The arms run sequentially, each with a two-update checkpoint followed by a fresh
process restoring model, both optimizer states, RNG and data cursor. Exact
validation agreement is required at restore. The Muon arm first runs an eight-rank
compiled BF16 check including a fully masked rank, identical rank updates and an
exact next-update optimizer restore. CPU unit tests run inside the pinned Torch
2.6 worker before each arm as well as locally.

The launcher retains local JSON after each completed arm and verifies equal data
identity, per-source exposures, token count and initial validation. Full metrics,
source snapshots, data manifest, checkpoints and logs live on the training volume.
It reports steady throughput excluding ten continuation steps, whole-run elapsed
time including both training processes, and peak GPU allocation. Container startup
and the separate qualification gate are outside that elapsed time. Validation is
one fixed window per source: diagnostic only. No quality superiority is assumed.

Changing optimizer/config/source rejects old resume checkpoints. This experiment
does not migrate, overwrite or publish the existing Simple LLM production model.

## Execution status — 2026-10-18

14 targeted local tests passed: 4 Muon tests, 7 distributed configuration tests,
and 3 existing training optimization tests. Local locked Torch is 2.14.1; the Modal
image remains Torch 2.6.0, matching the earlier speed experiment. GPU qualification
and training are **not verified**. The launch attempt could not connect to Modal:
`https://api.modal.com` returned HTTP 503 and a bounded client probe timed out.
No application ID or measured training results were returned. See the retained
[attempt record](../results/muon-124m-100mt-20261018-attempt.json).
