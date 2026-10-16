# Tokenizer learning curve

Compare 64, 128, 256, 512, 1,024 and 2,048 MiB of fitting text at 16,384,
32,768 and 50,000 vocabulary entries: 18 fresh candidates. The specification is
`learning-curve-v1.json`. Existing frozen `scholarly-code-v1` assets never change.

## Corpus

All 12 sources and their complete pinned shard lists in `../recipe-v1.json` are
eligible. “Full dataset” means the complete pinned source inventory, not a claim
that every upstream record is scanned. FineWeb remains the plan's 10BT subset.
The earlier 64 MiB tokenizer cache is not reused.

A seeded permutation orders shards per source; records stream sequentially.
Each document contributes at most 64 KiB and each parent at most 256 KiB. Source
weights apply to UTF-8 bytes. Each size takes nested per-source prefixes of the
frozen text. UTF-8 boundaries can leave at most three bytes unused per source.
This is one sampling seed, not a measurement of variance across training samples.

Parent hashing uses `scholarly-parent-v2`: 80% fitting, 10% evaluation, 10% excluded
test. A fixed, source-weighted 32 MiB evaluation corpus serves all candidates.
The original tokenizer's collector split was not retained, so its historical
metrics are not mixed into this matched experiment; 64 MiB candidates are new.

Exact whitespace-normalized chunks are deduplicated across the selected sample.
Cross-split collisions quarantine both copies, and quotas are rechecked before
freezing. Fuzzy near deduplication and cross-source parent identity resolution
remain limitations. Exhausted sources are not silently replaced or repeated.
SQLite transactions save cursors, chunks and decisions; failed collection resumes
with the same version and code. Manifests pin implementation, adapters, recipe
and corpus file hashes. Candidate output paths cannot be overwritten.

## Run on Modal

From `training/`:

```sh
uv sync --locked --extra corpus --extra research
uv run python -m unittest discover -s tests -v
uv run modal run --detach modal_tokenizer_research.py \
  --run-id learning-curve-20261016-v1 --phase all
```

In this managed environment, the SDK needs its proxy-support extra and system CA
trust. If it selects a separate CA bundle, use the installed truststore integration
without disabling TLS verification:

```sh
uv run python -c "import truststore,runpy; truststore.inject_into_ssl(); runpy.run_module('modal',run_name='__main__')" \
  run --detach modal_tokenizer_research.py --run-id learning-curve-20261016-v1 --phase all
```

The workflow runs remote tests, collects/freezes the corpus, then fits all 18
candidates with at most three concurrent workers. Collection uses four CPUs,
16 GiB RAM and a two-hour timeout. Each candidate uses eight CPUs, 64 GiB RAM,
a two-hour timeout, and no retries. A fresh subprocess isolates candidate peak
RSS. No GPU or ledger is required. Resource/time bounds are not billing estimates.

Phases: `smoke`, `collect`, `train`, `all`, or `candidate` with `--size-mib 64
--vocab-size 16384`. Inspect completed/partial results before retrying. Preserve
partial candidate directories before a deliberate new attempt. Detached mode
preserves active remote work, but the local coordinator must remain alive to
advance from collection to candidate dispatch and save returned reports.

Storage:

- `lm-exploration-mixture-data:/tokenizer-research/<run-id>/`: staging and `frozen/`.
- `lm-exploration-training:/tokenizer-research/<run-id>/<MiB>mib-<vocab>/`:
  `tokenizer.json` and full `result.json`, including parent-level totals.
- `results/tokenizer-learning-curve/<run-id>/`: downloaded measurements and report.

## Report

```sh
uv run python tokenizer_report.py results/tokenizer-learning-curve/<run-id>
```

Generates `REPORT.md`, `research-report.pdf`, compact `measurements.json`, and
Matplotlib PNG/SVG figures: aggregate and per-source learning curves, resource
use, vocabulary utilization and gains from additional fitting text. The report
compares 32K/50K with 16K at each size and reports paired, source-stratified parent
bootstrap intervals (256 replicates). Missing results stay explicitly pending.

Measurements include actual fitting bytes, actual vocabulary size, held-out
bytes/token, round-trip and UNK failures, vocabulary use/unseen entries, fixed
technical/code/math fragmentation probes, fitting time, peak process RSS, and
evaluation throughput. Throughput includes decoding and measurement overhead.
Intervals describe evaluation-sample uncertainty, not seed variance or model
quality. Check all domains and embedding/output parameter costs before choosing
a tokenizer. Matched model-compute comparisons using bits per byte are later work.
