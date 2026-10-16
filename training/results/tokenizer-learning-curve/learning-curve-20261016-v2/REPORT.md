# Tokenizer learning-curve research report

## Overview

0 of 18 requested candidates have measured Modal results. The grid doubles fitting text from 64 MiB through 2 GiB and compares 16,384, 32,768 and 50,000 vocabulary entries. All fresh candidates use the same source-weighted, parent-held-out evaluation corpus. A higher bytes-per-token value means fewer tokens for the same text; it does not establish better language-model quality.

**Status: in progress — incomplete matrix.**

## Experimental design

- All 12 pinned source inventories from the scholarly/code plan are eligible; the experiment samples them rather than claiming to process every upstream record.
- Nested per-source UTF-8 byte prefixes: 64, 128, 256, 512, 1,024 and 2,048 MiB. 16K means 16,384 entries, 32K means 32,768, and 50K means exactly 50,000.
- Source weights follow the plan, applied to bytes for tokenizer fitting. This is distinct from the eventual model-training token mixture.
- Shared 32 MiB evaluation corpus; parent-level train/evaluation/test separation before chunking. Test parents are excluded.
- Byte-level BPE, full byte alphabet, no normalization or automatic boundaries; the frozen v1 pretokenization rules are held fixed. The original tokenizer is unchanged.
- Seeded shard permutation, sequential records within shards, a 64 KiB document cap and 256 KiB parent cap. Exact whitespace-normalized duplicate chunks are removed; cross-split conflicts are quarantined before freezing.
- CPU-only Modal: eight CPUs, 64 GiB memory limit and two-hour candidate timeout; at most three candidates concurrently. Peak RSS includes fitting and evaluation. Evaluation throughput includes encoding, decoding, counting and checks.

## Aggregate learning curve

![Learning curve](learning-curve.png)

Shading is a 95% parent-bootstrap interval, stratified by source with fixed source-byte weights (256 paired replicates). It describes evaluation-sample uncertainty, not uncertainty across training seeds.

| MiB | Vocabulary requested / actual | Bytes/token | Token reduction vs 64 MiB | Fit seconds | Peak RSS MiB | Correctness |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 16,384 / — | pending | — | — | — | pending |
| 64 | 32,768 / — | pending | — | — | — | pending |
| 64 | 50,000 / — | pending | — | — | — | pending |
| 128 | 16,384 / — | pending | — | — | — | pending |
| 128 | 32,768 / — | pending | — | — | — | pending |
| 128 | 50,000 / — | pending | — | — | — | pending |
| 256 | 16,384 / — | pending | — | — | — | pending |
| 256 | 32,768 / — | pending | — | — | — | pending |
| 256 | 50,000 / — | pending | — | — | — | pending |
| 512 | 16,384 / — | pending | — | — | — | pending |
| 512 | 32,768 / — | pending | — | — | — | pending |
| 512 | 50,000 / — | pending | — | — | — | pending |
| 1024 | 16,384 / — | pending | — | — | — | pending |
| 1024 | 32,768 / — | pending | — | — | — | pending |
| 1024 | 50,000 / — | pending | — | — | — | pending |
| 2048 | 16,384 / — | pending | — | — | — | pending |
| 2048 | 32,768 / — | pending | — | — | — | pending |
| 2048 | 50,000 / — | pending | — | — | — | pending |

## Interpretation and limitations

Do not select a winner from an incomplete matrix. Once complete, examine same-vocabulary gains from each doubling, source regressions, fragmentation probes and resource use together. A compression plateau is evidence of diminishing tokenizer-fitting returns, not proof of downstream optimality.

This experiment uses one corpus sampling seed. It does not establish stability across independent corpus samples. Near duplicates with changed wording or shifted chunk boundaries and equivalent books/papers across sources may remain. These limitations must be resolved for strong generalization claims. The frozen v1 collector split is not reproducible from its handoff, so its historical results are not mixed into this matched learning curve.

Token-level language-model perplexity is not compared here. Downstream evaluation should use the same held-out text and bits per byte under matched model size and compute. No model quality or billing-cost improvement is claimed.

## Reproduction and evidence

- `measurements.json` contains compact metrics, per-source comparisons, fragmentation probes and raw-result hashes. Full parent totals and candidate tokenizers remain on the Modal results volume.
- `corpus-manifest.json` records source pins, sampling settings, corpus hashes and collection decisions; `modal-smoke.json` records remote test execution.
- Run `python tokenizer_report.py <results-folder>` to regenerate this report and Matplotlib PNG/SVG/PDF figures from downloaded raw result JSONs.
