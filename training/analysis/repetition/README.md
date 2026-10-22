# Repetition analysis

This study runs low-temperature sampling on four published Simple LLM 124M / 8K
checkpoints. It makes no training updates. Model code and weights load from pinned
revisions of `jfrick/scholarly-code-modern-gpt-124m-step2`, not from the current
repository architecture.

## Findings

Repeated token 4-gram fraction, averaged over the same ten prompts with a 128-token cap.
Sampled settings use top-k 50 with two seeds per prompt. Lower repetition does not
establish factual correctness.

| Trained tokens | Held-out loss¹ | Greedy | T=0.1 | T=0.2 | T=0.4 | T=0.8¹ |
|---|---:|---:|---:|---:|---:|---:|
| 0.077B | 3.9120 | 83.8% | 83.7% | 84.2% | 70.6% | 20.2% |
| 1.245B | 2.2869 | 83.9% | 80.5% | 80.8% | 67.1% | 34.3% |
| 2.491B | 2.1151 | 80.9% | 81.9% | 78.0% | 53.4% | 24.4% |
| 5.000B | 1.9635 | 77.1% | 75.7% | 73.8% | 66.1% | 21.4% |

¹ Held-out loss and T=0.8 come from the earlier matched-checkpoint comparison: 1,798,432
identical targets from 384 validation parents. That evaluation is exploratory, not a sealed
benchmark. Its final loss differs from the 1.8708 training-time diagnostic because it uses
different samples.

- Very low temperatures did not resolve repetition. T=0.1 and T=0.2 stayed close to greedy
  at every checkpoint. T=0.4 reduced repetition but stayed much more repetitive than T=0.8.
- No repetition penalty or n-gram ban was used. All 280 completions are retained.
- Greedy reproducibility: all 40 greedy completions match the earlier comparison token for
  token. Both runs used PyTorch 2.6 on an H100 with BF16 autocast and FP32 weights.
- The earlier study's training checks are archived with it and found no simple
  implementation error. They covered 280 real windows (alignment, EOS, document boundaries,
  padding), cursor replay, loader/prefetch tests and a two-rank FP32 gradient/AdamW reference.
  They do not prove the pipeline is bug-free.

## Archived outputs

Everything lives in the model repo
[`jfrick/scholarly-code-modern-gpt-124m-step2`, folder `analysis/repetition/`](https://huggingface.co/jfrick/scholarly-code-modern-gpt-124m-step2/tree/main/analysis/repetition):
report, plots, completions, raw results (`low-temperature-20261018.json`,
`previous-comparison-20261018.json`), `summary.json`, the data audit and implementation checks.

## Rerun

Use the [generic toolkit](../sampling/README.md) from `training/`. The
[`low-temperature.json`](low-temperature.json) recipe reproduces this study.
[`previous-comparison.json`](previous-comparison.json) is the generation part of the earlier
comparison; its held-out loss comes from `modal_scripts/modal_validate_repetition.py`.

```sh
modal run --detach -m sampling.modal_sample --recipe analysis/repetition/low-temperature.json \
  --output artifacts/repetition/low-temperature.json
```

To rebuild the table from the archived raw inputs, which use the earlier per-study result
format that the current report no longer reads, run these commands from a checkout of
revision `1e06fb8`:

```sh
uv run hf download jfrick/scholarly-code-modern-gpt-124m-step2 \
  analysis/repetition/low-temperature-20261018.json analysis/repetition/previous-comparison-20261018.json \
  --local-dir artifacts/hf
A=artifacts/hf/analysis/repetition
uv run python -m sampling.report $A/low-temperature-20261018.json \
  --reference $A/previous-comparison-20261018.json --group-by temperature \
  --output-dir artifacts/repetition/low-temperature
uv run python -m sampling.report $A/previous-comparison-20261018.json --group-by decoding \
  --output-dir artifacts/repetition/previous
```

These commands reproduce every value in the archived `summary.json`. That covers 16 mean
repetitions, the four T=0.8 means, 280 completions and 40/40 identical greedy completions,
to within 1e-12 relative. Five values differ only in the last bit, because the archived
summary averaged with `sum/len`.
