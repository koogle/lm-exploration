# Experiment evidence

Current priority: [DFlash-style fitting results and curves](dflash-findings.md).
The baseline fits its tiny training corpus but fails to generalize; no speedup.

## Verified initial checks (2026-10-11)

- `smoke-l40s-01.json`: seven variants, 10 updates each, 64 training windows,
  16 held-out windows, one J-lens calibration prefix, 3.66M trainable parameters.
  About 51.8 seconds elapsed and 1.45 GB peak allocated GPU memory. Validates the
  original pipeline; **not a useful model-quality comparison**. This older run's
  generation timing also computed J-lens features for raw-state variants, so do
  not use those latency columns as conditioning-specific performance evidence.
- `profile-l40s-01.json`: 21.63M parameters, 25 steps, 6.64 steps/sec, 1.46 GB peak.
  Warm block decoding 7.38 ms/token, AR 12.14 ms/token on the same prompt.
- `profile-h100-01.json`: identical configuration, 5.47 steps/sec, 1.51 GB peak.
  Warm block decoding 12.01 ms/token, AR 16.13 ms/token.

Both profiles are tiny single-container observations with timing noise and Python
orchestration overhead. They support selecting L40S for this pilot, not a general
hardware ranking or matched-quality speedup. The decoder is poorly trained here.

## Completed exploratory pilot

`pilot-l40s-01`: 21.63M-parameter decoder, 100 training-prefix J-lens calibration,
selected layers 6/12/18, seven conditions, 750 updates × batch 8 each, 256-token
contexts, four-token targets. Actual exposures: 1,536,000 context tokens and
24,000 target tokens per condition; 10,752,000 and 168,000 across seven conditions.
These are repeated exposures; token windows are sampled from 4096 training
examples. The [measured report](pilot-l40s-01.md) shows repeated punctuation and poor
held-out quality. This does not validate any conditioning hypothesis.

Use `uv run python scripts/summarize.py results/RUN.json --output results/RUN.md`
for a readable table. Raw JSON retains per-position and per-example metrics.

## Mixed-domain data gate

`mixed-smoke-01.json` completed on L40S with the updated output normalization,
mixed prose/code/translation training and disjoint train/validation/test splits.
All three paths (final, all extracted layers, J-lens) completed five updates;
test metrics remain sealed. This is plumbing verification only.

`data-study-01` compares nested data subsets at equal optimization checkpoints.
[Measured results and curves](data-study-01.md): validation NLL at 2k updates
is 9.65/7.80/7.52 for 768/3,072/12,288 windows. Generation remains repetitive;
test remains sealed. See [protocol](../docs/data-budget.md).


## Clean follow-up and diagnostics

See [consolidated measurements](data-scaling.md) and [curves](data-scaling.png).
The original data-study-01 pool contained one validation-overlapping window
(nine training copies) and duplicate training windows. Its numbers are historical
exploration, not clean validation evidence. The repaired studies remove these
rows, refill balanced nested subsets, and retain unchanged held-out tensors.

`data-position-memorize-01` establishes tiny-set trainability, not generalization.
`data-repaired-02`, `data-large-01`, `data-batch128-01` and
`data-large-batch128-01` separate available unique data and optimization exposure.
`data-final-control-01`, `data-normalized-01` and `data-posthoc-batch-01` are
negative quality diagnostics. Large raw reports use lossless `.json.gz` archives.


## Completed conditioning study

[Findings](conditioning-findings.md), [training curves](conditioning-report.md),
[two-seed layer scan](conditioning-layer-scan.png), [Qwen3 comparison](qwen3-conditioning-report.md)
and [frozen test evaluation](conditioning-test.md) cover the actual trained study.
All 11 checkpoint selections were frozen before test access; the test is now
exposed. The source report hashes, checkpoint hashes and corpus identity audit
are retained. Best heads and resumable checkpoints are in the dedicated Modal
volume. The all-layer Qwen2.5 inference head is also downloaded locally.
