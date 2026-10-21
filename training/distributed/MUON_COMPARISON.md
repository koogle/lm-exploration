# Fresh 124.4M Muon versus AdamW comparison

**Completed October 19.** Muon diagnostic loss was 2.7204 versus AdamW 3.5525 at
matched 99.99M tokens, with 9.8% lower steady throughput. Both GPU runs and restore
checks passed. The measured report, combined summary, per-step metrics, validation
histories, curve and the failed first-attempt record are archived in the HF dataset
`jfrick/lm-exploration-results` under [`pr-24/training/results/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-24/training/results)
(`muon-124m-100mt-*`).

## Measured results (run prefix `muon-124m-100mt-20261019-b`)

Both arms: 99,988,281 nonpadding tokens / 1,296 updates, seed 42, Modal app
`ap-Pqtzos5cN2U8xxPqAj5UKc`, implementation commit `ebc5464`.

| Measurement | AdamW | Muon + AdamW |
| --- | ---: | ---: |
| Final weighted diagnostic loss (nats) | 3.5525 | 2.7204 |
| Steady nonpadding tokens/sec | 275,395 | 248,500 |
| Peak allocated GiB (max over ranks) | 14.364 | 13.993 |
| Combined training-process seconds | 734.18 | 642.39 |
| Fresh-process restore validation | exact | exact |

Muon is 0.8322 nats (23.4%) lower and had lower loss on all 12 per-source diagnostic
windows (e.g. code_python 2.4095 vs 3.5364, fineweb_edu 3.9616 vs 4.5927). Its steady
throughput is 9.8% **lower**; the shorter combined time is not a speed win (AdamW ran
first and the arms shared compiler caches, so startup was not a controlled cold-cache
comparison). One seed, untuned LRs and one fixed validation window per source:
diagnostic evidence only, not long-run quality. Next: larger held-out set, repeated
seeds/LRs, and profiling the uncompiled Muon matrix updates.

To rebuild the curve, download the `*-validation.jsonl` files into an ignored folder
and run `uv run --with matplotlib python analysis/muon/plot_comparison.py --input-dir
<folder> --output artifacts/pr-24/muon-curve.png` from `training/`.

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
uv run modal run -m modal_scripts.modal_muon_comparison --prefix muon-124m-100mt-<fresh-id>
```

The arms run sequentially, each with a two-update checkpoint followed by a fresh
process restoring model, both optimizer states, RNG and data cursor. Exact
validation agreement is required at restore. The Muon arm first runs an eight-rank
compiled BF16 check including a fully masked rank, identical rank updates and an
exact next-update optimizer restore. CPU unit tests run inside the pinned Torch
2.6 worker before each arm as well as locally.

The launcher retains local JSON in the ignored `artifacts/pr-24/results/` (or
`--output-dir`) after each completed arm and verifies equal data
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
No application ID or measured training results were returned. The attempt record
(`muon-124m-100mt-20261018-attempt.json`) is archived in the HF folder above.

## Proxy certificate setup

If the Modal client reports `CERTIFICATE_VERIFY_FAILED` behind the managed
workspace proxy, use the installed `truststore` package to load system certificate
authorities while keeping TLS verification enabled:

```sh
uv run python -c 'import truststore; truststore.inject_into_ssl(); from modal.__main__ import main; main()' run -m modal_scripts.modal_muon_comparison --prefix muon-124m-100mt-<fresh-id>
```

On October 19 this resolved the client connection failure. The retry was accepted
as Modal app `ap-Pqtzos5cN2U8xxPqAj5UKc`, prefix
`muon-124m-100mt-20261019-b`. Both arms subsequently completed; see the measured results above.
