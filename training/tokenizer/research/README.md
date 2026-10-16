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

The initial 8 KiB evaluation cap exhausted LibreTexts at 636,017 bytes; the
retained failed attempt documents why the final cap is 16 KiB.

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
  --run-id learning-curve-20261016-v2 --phase all
```

In this managed environment, the SDK needs its proxy-support extra and system CA
trust. If it selects a separate CA bundle, use the installed truststore integration
without disabling TLS verification:

```sh
uv run python -c "import truststore,runpy; truststore.inject_into_ssl(); runpy.run_module('modal',run_name='__main__')" \
  run --detach modal_tokenizer_research.py --run-id learning-curve-20261016-v2 --phase all
```

The workflow runs remote tests, collects/freezes the corpus, then fits all 18
candidates with at most three concurrent workers. Collection requests four CPUs and
16 GiB RAM with a two-hour timeout. Each candidate requests eight CPUs and 64 GiB RAM,
a two-hour timeout, and no retries. A fresh subprocess isolates candidate peak
RSS. No GPU or ledger is required. CPU/memory settings are requests, not hard caps; candidate Rayon threads are set to eight.
Concurrency and timeout bounds are enforced. Resource settings are not billing estimates.

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

## Broader held-out parent coverage

The first corpus audit found only five LibreTexts evaluation parents. Before
interpreting curves, we kept the fitting corpus and all tokenizer weights fixed
and rebuilt evaluation with a 16 KiB document/parent cap, retaining the same
32 MiB source quotas and parent split. A separate dedup index reserves all selected
fitting chunk hashes. This broadens coverage but does not fix near duplicates or
cross-source parent identity. Original evaluation metrics remain on Modal.

```sh
uv run modal run --detach modal_tokenizer_evaluation.py \
  --run-id learning-curve-20261016-v2 --phase collect
# After collection and all 18 fits have finished:
uv run modal run --detach modal_tokenizer_evaluation.py \
  --run-id learning-curve-20261016-v2 --phase evaluate
uv run python tokenizer_report.py results/tokenizer-learning-curve/learning-curve-20261016-v2
```

Collection has four CPU/16 GiB requests and hard caps and a two-hour timeout.
Rescoring has two CPU/8 GiB requests and hard caps, three concurrent containers,
and a 20-minute timeout. The evaluation manifest is downloaded alongside the
original fitting manifest. `result-balanced-evaluation.json` is saved next to
original `result.json` on the results volume. Local raw results are replaced only
after all 18 rescoring jobs return successfully. Report generation rejects mixed
evaluation hashes. Fitting time and peak RSS retain their original measurements;
evaluation throughput is measured in the new two-CPU rescoring workers.

## Hugging Face publication

All 18 measured candidates are packaged in
[`jfrick/scholarly-code-tokenizer`](https://huggingface.co/jfrick/scholarly-code-tokenizer).
The default is 2 GiB / 32K, balancing compression and vocabulary parameter overhead;
this is a starting point, not a downstream model-quality optimum. Each alternative
is available through `subfolder="variants/<MiB>mib-<vocab>"` in `AutoTokenizer`.
The existing frozen v1 assets in this repository remain unchanged.

After downloading candidate `tokenizer.json` files into
`<artifacts>/variants/<MiB>mib-<vocab>/`, build a fresh publication directory:

```sh
uv run python package_tokenizer_research.py \
  results/tokenizer-learning-curve/learning-curve-20261016-v2 \
  <artifacts> <new-publication-folder> --repo-id jfrick/scholarly-code-tokenizer
```

Packaging verifies all candidate hashes and the shared audited evaluation manifest.
It includes standard Hugging Face tokenizer configuration, all 18 variants, a model
card, the report/figures and provenance. No raw dataset text or model weights are
included. Standard loading was checked with `transformers==4.57.6` and
`tokenizers==0.22.2` in an isolated environment. Publication uses the authenticated
Hugging Face account; credentials are never stored in the artifacts.

To verify a downloaded publication, use an isolated Transformers environment
(the older Transformers Hub dependency differs from the corpus collector):

```sh
uv run --isolated --no-project --with transformers==4.57.6 \
  --with tokenizers==0.22.2 python verify_tokenizer_publication.py <publication-folder>
# Or check the published repository at its immutable commit:
uv run --isolated --no-project --with transformers==4.57.6 \
  --with tokenizers==0.22.2 python verify_tokenizer_publication.py \
  jfrick/scholarly-code-tokenizer --revision <commit-sha>
```

## JSON, code, diffs and tool calls

The [structured-output follow-up](../../results/tokenizer-structured/structured-code-v1/REPORT.md)
scores all 18 original candidates and three fresh 64 MiB code-focused fits at
16K/32K/50K. It uses 512 disjoint held-out Python parents to construct each of four
workloads. JSON, patches and tool formats are controlled generated fixtures, not
independent production traces. The focused mixture is 40/20/20/20 by UTF-8 bytes;
all new candidates are also checked on the original scholarly mixture.

Run `modal run --detach modal_structured_tokenizer.py`, then
`python structured_tokenizer_report.py` from `training/`. The Modal entrypoint
runs two remote tests first and uses explicit CPU/memory hard caps, concurrency
limits and timeouts. Existing artifact paths refuse overwrite; use a fresh run
version for a new collection. Raw per-parent scores remain under
`runs/structured-code-v1/` locally and on the results volume. The generated report
retains compact metrics, paired bootstrap intervals and provenance. Original
published tokenizers and the default remain unchanged.
