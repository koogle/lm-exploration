# Simple LLM 124M: decoding presets

This bounded inference experiment compares decoding settings on the final 5B-token
checkpoint. It does not train, change weights, or publish to the model repository.

Model: `jfrick/scholarly-code-modern-gpt-124m-step2`, revision
`38df193c2cc2c5463e4bb20a5c6a4998f58b3af0` (step 64,227;
4,999,981,176 trained targets; 124,389,632 parameters). The program loads the exact
published model definition from that revision, verifies source and weight hashes,
and checks finite parameters. Current repository model code is intentionally not
used because the architecture has changed since training.

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

**No setting fixed general generation quality.** Manual inspection found readable
phrases alongside contradictions, topic drift, and malformed or incorrect code.
The [paired examples](quality-review.md) include counterexamples where a penalty
reduced repetition but damaged an initially useful answer. No subjective quality
percentage or automated judge score is reported.

For prose exploration, **T=.85, top-p=.95, repetition_penalty=1.1** is a provisional
anti-loop candidate: it substantially reduces loops without choosing the highest
tested temperature. It is not a validated fluency or relevance improvement.
T=1/penalty1.1 minimizes this study's repetition metric but is not recommended
solely on that basis. For code, this experiment identifies **no reliable preset**;
retain a penalty-1.0 control because discouraging necessary repeated tokens can
damage the program. A fresh prompt set with more seeds is needed before promoting
any setting to a general default. No product default is changed by this PR.

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
the prompt or generated prefix. Each identity is penalized once, regardless of
its occurrence count. At penalty 1.0 this is a no-op. Repeated subword tokens are
penalized too: this can damage identifiers, indentation, and normal language.

Repetition is `1 - unique_generated_token_4grams / all_generated_token_4grams`,
averaged equally across completions. Prompt tokens are excluded from that metric.
The report also separates the seven prose/math prompts from the three Python
prompts and reports repeated character 16-grams over the first 256 generated
characters. EOS and completion lengths are retained to expose early termination.
Neither metric establishes relevance, correctness, fluency, or executable code.
The summary additionally includes lowercased, whitespace-collapsed character
16-grams over the first 256 normalized characters, explicitly named separately
from the raw-character metric for comparison with the external-baseline study.

## Reproduce

Run from this directory with Modal credentials and access to the existing
`lm-exploration-training` output volume:

```sh
modal run --detach modal_decoding.py
python build_report.py
```

The job uses one H100, four requested CPUs, 8 GiB RAM, a 30-minute hard timeout,
and no automatic function retries. Downloads have four bounded retry attempts.
Completed matching results are reused from the output volume. The remote image
pins PyTorch 2.6.0 and tokenizers 0.22.2; model execution uses FP32 weights with
BF16 autocast. The local report builder uses only Python's standard library.

For a local environment whose outbound proxy uses the system CA store, the
equivalent launch command used here is:

```sh
python -c 'import truststore; truststore.inject_into_ssl(); from modal.__main__ import main; main()' run --detach modal_decoding.py
```

TLS verification remains enabled. No credentials, local model weights, or
generated code execution are required in this folder.

## Execution and artifacts

Completed Modal app:
[ap-ZMqpEOdumyv0gGOiTsl7aT](https://modal.com/apps/radiantai/main/ap-ZMqpEOdumyv0gGOiTsl7aT).
Remote execution, including model download/loading and inference, took
188.822 seconds on one H100. At the previously checked Modal H100 rate of
$3.9492/GPU-hour, that is approximately **$0.21 of GPU time**, plus CPU/RAM and
startup charges. This is a rate-based estimate, not an invoice. No additional
training, deployment, or model-repository upload occurred.

Validation: source/weight provenance checks passed remotely; all 170 sample
records and all twenty matching prior-study baselines were checked locally.
Sign-aware penalty and nucleus threshold-crossing retention passed CPU reference
checks; both Python files compile. The report builder regenerates the summary,
table, baseline comparisons, and unfiltered completion document from raw JSON.

[Raw results](decoding-presets-20261018.json) · [Summary](summary.json) ·
[All completions](all-completions.md) · [Qualitative review](quality-review.md) ·
[Baseline reproducibility](baseline-reproducibility.json)

## Full-FP32 precision sensitivity

The independent external-baseline study found substantial BF16 sensitivity in
Pythia. That motivated an additional controlled check on **our** model, preserving
our original BOS prefix and matching prompts, seeds, and decode settings. It uses
full-FP32 weights and forward execution, CUDA/cuDNN TF32 disabled, and highest
float32 matmul precision. Original BF16 artifacts remain unchanged.

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
insensitive to inference precision.

The T=1 comparison uses only the ten matching original seeds, explaining its
6.32% BF16 baseline versus the twenty-completion sweep's 6.24% mean.

The additional job completed 80 samples in 99.215 seconds of remote execution
(approximately $0.11 GPU time plus CPU/RAM, excluding any unbilled queue time;
not an invoice):
[ap-D4K27UYN5JP4NSmEMulsi7](https://modal.com/apps/radiantai/main/ap-D4K27UYN5JP4NSmEMulsi7).
The raw artifact preserves an original copied descriptive note and explicitly
records its correction; actual per-mode top-k/nucleus fields and all generated
sample data are unchanged.

```sh
modal run --detach modal_precision.py
python build_precision_report.py
```

[Raw FP32 results and token IDs](own-precision-fp32-20261018.json) ·
[Matched precision comparisons and text](precision-comparison.json) ·
[Generated comparison table](precision-results.md)
