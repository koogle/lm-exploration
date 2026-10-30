# Simple LLM 124M: decoding presets

This bounded inference experiment compares decoding settings on the final 5B-token
checkpoint. It does not train, change weights, or publish to the model repository.

Model: `jfrick/scholarly-code-modern-gpt-124m-step2`, revision
`38df193c2cc2c5463e4bb20a5c6a4998f58b3af0` (step 64,227;
4,999,981,176 trained targets; 124,389,632 parameters). The program loads the exact
published model definition from that revision, verifies source and weight hashes,
and checks finite parameters. Current repository model code is intentionally not
used because the architecture has changed since training.

Generated outputs (raw JSONs with token IDs, summary tables, all 170 completions,
the FP32 comparison and the manual qualitative review) are archived on Hugging Face in
[`jfrick/lm-exploration-results` under `pr-20/training/analysis/decoding-presets/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-20/training/analysis/decoding-presets)
(private dataset).

## Measured results

| Preset | Temperature | Top-p | Penalty | Repeated token 4-grams | Prose/math | Python |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 0 | — | 1.00 | 77.1% | 82.5% | 64.5% |
| Sampling baseline | 1.00 | .95 | 1.00 | 6.2% | 4.7% | 9.9% |
| Lower temperature | .85 | .95 | 1.00 | 20.7% | 16.6% | 30.1% |
| .70 with penalty | .70 | .95 | 1.10 | 12.7% | 11.5% | 15.5% |
| .85 with penalty | .85 | .95 | 1.10 | 3.5% | 1.8% | 7.5% |
| 1.00 with penalty | 1.00 | .95 | 1.10 | 1.4% | 1.8% | 0.7% |
| Narrower nucleus, mild penalty | .85 | .90 | 1.05 | 18.0% | 11.5% | 33.1% |
| Narrower nucleus, stronger penalty | .85 | .90 | 1.15 | 3.9% | 5.0% | 1.3% |
| .70, narrower nucleus | .70 | .90 | 1.10 | 19.4% | 16.7% | 25.6% |

All ten greedy outputs and all ten shared-seed T=1/top-p=.95 baseline outputs
match the previous study **token for token**. The new sampling baseline mean
uses twenty completions, explaining its 6.24% versus the previous ten-sample
6.32%. All but three of the 170 completions reached the 128-token cap; two EOS
completions occur at T=1/penalty1.1, one at T=.85/top-p=.90/penalty1.15.

The controlled T=.85/top-p=.95 pair shows a penalty of 1.1 reducing mean token
repetition from 20.68% to 3.48%. The T=1/top-p=.95 pair decreases from 6.24% to
1.44%. A tighter nucleus did not uniformly help: at T=.70/penalty1.1, top-p=.90
was more repetitive than top-p=.95 (19.36% versus 12.72%). These are observations
on a small reused prompt set, not population estimates.

**No setting fixed general generation quality.** A manual review (first 240
characters of all 90 first-seed completions, plus full second-seed completions for
the baseline and the three lowest-repetition presets; no generated code executed,
no automated judge or quality percentage) found readable phrases alongside
contradictions, topic drift, invented names and malformed or incorrect code.
Counterexamples from that review:

- Breaking a loop is not coherence: for `The solar system consists of`, greedy
  repeats "a solar system"; T=.85/p=.95/penalty1.1 avoids the loop but drifts to
  cameras, a power station and heat transfer.
- A penalty can spoil a useful start: for `2 + 2 =` (seed 1008, T=1/p=.95) penalty
  1.0 starts with 4 and penalty 1.1 with 3; seed 2008 reverses the direction. This
  is a tradeoff example, not an accuracy estimate.
- Stronger penalties can damage code: for `def add(a, b):` (seed 1004,
  T=.85/p=.90) penalty 1.05 writes `a += b; return a`, while penalty 1.15 increments
  both arguments before adding them.

For prose exploration, **T=.85, top-p=.95, repetition_penalty=1.1** is a provisional
anti-loop candidate: it substantially reduces loops without choosing the highest
tested temperature. Keep the unpenalized T=1/top-p=.95 control beside it. It is
not a validated fluency or relevance improvement. T=1/penalty1.1 minimizes this
study's repetition metric but often produces especially disordered prose or code,
so it is not recommended on that basis. For code, this experiment identifies
**no reliable preset**; retain a penalty-1.0 control because discouraging
necessary repeated tokens can damage the program. A task-specific stop rule might
preserve useful prefixes but was not tested. A fresh prompt set with more seeds
is needed before promoting any setting to a general default. No product default
is changed by this PR.

## Full-FP32 precision sensitivity

The independent external-baseline study found substantial BF16 sensitivity in
Pythia. This check reruns **our** model with the same BOS prefix, prompts, seeds
and decode settings, using full-FP32 weights and forward execution, CUDA/cuDNN
TF32 disabled and highest float32 matmul precision.

| Setting | Matched completions | BF16 repeat token4 | FP32 repeat token4 | Token-identical completions |
|---|---:|---:|---:|---:|
| Greedy | 10 | 77.12% | 77.20% | 9 |
| T=.1, top-k=50 | 20 | 75.72% | 74.92% | 14 |
| T=.8, top-k=50 | 20 | 21.36% | 22.88% | 1 |
| T=1, top-p=.95 | 10 | 6.32% | 4.32% | 0 |
| T=.85, top-p=.95, penalty1.1 | 20 | 3.48% | 4.68% | 0 |

**Full FP32 did not fix our repetition.** Nine of ten greedy completions are
token-identical. The remaining France completion switches from repeating
“the capital of the French Empire” to repeating “the capital of the United
States.” Both are wrong and highly repetitive. Sampled paths are more sensitive
to small numerical differences, but the broad repetition pattern persists.
This result is specific to our model; do not infer that other models are
insensitive to inference precision. The T=1 row uses only the ten matching
original seeds (6.32% BF16 versus the sweep's twenty-completion 6.24%). The raw
FP32 artifact records a corrected descriptive note; sample data are unchanged.

## Method

Ten fixed prompts from the previous repetition study, including three Python
prompts, are evaluated at a 128-new-token limit. Greedy has one completion per
prompt. Eight sampled presets each use seeds `1000 + prompt_index` and
`2000 + prompt_index`: 170 completions in total. Same prompts and seed identities
across settings permit comparison; two seeds are insufficient to estimate broad
generation quality or confidence intervals. These prompts are reused diagnostics,
not a fresh held-out benchmark.

Nucleus sampling keeps the token that crosses the top-p threshold. No top-k
filter or n-gram ban is applied. The sign-aware repetition penalty divides
positive logits and multiplies negative logits for token identities already in
the prompt or generated prefix, once per identity regardless of count; 1.0 is a
no-op. Repeated subword tokens are penalized too, which can damage identifiers,
indentation and normal language.

Repetition is `1 - unique_generated_token_4grams / all_generated_token_4grams`,
averaged equally across completions, excluding prompt tokens. The summary also
splits prose/math from Python prompts. It reports word 4-grams and normalized
(lowercased, whitespace-collapsed) character 16-grams, plus EOS counts and completion
lengths. Neither metric establishes
relevance, correctness, fluency or executable code.

## Reproduce

The two recipes here run on the [generic toolkit](../sampling/README.md):
[`decoding-presets.json`](decoding-presets.json) for the sweep and
[`precision-fp32.json`](precision-fp32.json) for the FP32 check. Run them from `training/`
with Modal credentials and the `lm-exploration-training` volume. Each launch uses one H100,
a 30-minute hard timeout and no retries. The image pins PyTorch 2.14.1 and tokenizers 0.22.2.

```sh
modal run --detach -m sampling.modal_sample --recipe analysis/decoding-presets/decoding-presets.json \
  --output artifacts/decoding-presets/decoding-presets.json
modal run --detach -m sampling.modal_sample --recipe analysis/decoding-presets/precision-fp32.json \
  --output artifacts/decoding-presets/precision-fp32.json
```

To rebuild the tables, download the raw outputs archived in
[`jfrick/lm-exploration-results`, folder `pr-20/training/analysis/decoding-presets/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-20/training/analysis/decoding-presets).
The two repetition-study files come from that study's
[archive](../repetition/README.md#archived-outputs). These inputs use the earlier
per-study result format, which the current report no longer reads; run the commands
from a checkout of revision `1e06fb8`.

```sh
uv run hf download jfrick/lm-exploration-results --repo-type dataset \
  --local-dir artifacts/hf \
  --include 'pr-20/training/analysis/decoding-presets/*.json'
P=artifacts/hf/pr-20/training/analysis/decoding-presets R=artifacts/hf/analysis/repetition
uv run python -m sampling.report $P/decoding-presets-20261018.json \
  --reference $R/previous-comparison-20261018.json \
  --recipe analysis/decoding-presets/decoding-presets.json \
  --group-by mode --group-by mode,prompt_group --output-dir artifacts/decoding-presets/sweep
uv run python -m sampling.report $P/own-precision-fp32-20261018.json \
  --reference $P/decoding-presets-20261018.json --reference $R/low-temperature-20261018.json \
  --reference $R/previous-comparison-20261018.json --output-dir artifacts/decoding-presets/fp32
```

These commands reproduce the following archived values exactly:

- every per-preset mean token repetition, including the prose/Python split;
- completion, EOS and Python-completion counts, and mean completion lengths;
- all 20 baseline reproducibility pairs (20 identical);
- every FP32 comparison value: per-mode BF16/FP32 means, token-identical counts, and
  per-sample first-divergence positions.

The archived summary's normalized character 16-gram value over the first 256 characters
included completions shorter than 256 characters. The toolkit's `char_rep16_first256` leaves
those out and reports its coverage, so its values differ. The tables above do not use that
metric.

## Execution

- Sweep: [ap-ZMqpEOdumyv0gGOiTsl7aT](https://modal.com/apps/radiantai/main/ap-ZMqpEOdumyv0gGOiTsl7aT),
  188.822 s remote execution on one H100 (about $0.21 GPU time at $3.9492/GPU-hour,
  plus CPU/RAM and startup; not an invoice).
- FP32 check: [ap-D4K27UYN5JP4NSmEMulsi7](https://modal.com/apps/radiantai/main/ap-D4K27UYN5JP4NSmEMulsi7),
  80 samples in 99.215 s (about $0.11 GPU time).

Validation: source/weight provenance checks passed remotely; all 170 sample
records and all twenty matching prior-study baselines were checked locally.
Sign-aware penalty and nucleus threshold-crossing retention passed CPU reference
checks. No training, deployment or model-repository upload occurred.
