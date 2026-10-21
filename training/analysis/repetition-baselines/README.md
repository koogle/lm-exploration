# Simple LLM repetition: external baselines and corpus patterns

Bounded diagnostic comparing the final Simple LLM 124M / 5B checkpoint with base models
Pythia-160M (step 2000, about 4.19B tokens; final, 299.89B tokens) and SmolLM-135M (600B
tokens), plus a read-only audit of the training corpus. No training updates.

Raw outputs, all completions, metrics tables, the corpus audit and manual notes are on
Hugging Face in
[`jfrick/lm-exploration-results` under `pr-23/training/analysis/repetition-baselines/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-23/training/analysis/repetition-baselines)
(private dataset; its `pr-23/README.md` lists the files).

This folder holds only the study: recipes for the shared
[sampling toolkit](../sampling/README.md), two audit scripts the toolkit does not cover, and
these findings.

| File | Role |
|---|---|
| `external-fp32.json` | Recipe: Pythia/SmolLM baselines, full FP32, tokenizer-default prefixes (primary) |
| `external-bf16.json` | Recipe: initial BF16-autocast run, Pythia with EOS-as-BOS (token 0) prepended |
| `modal_corpus_patterns.py` | Read-only training-corpus audit (CPU, read-only SQLite) |
| `modal_verify_pythia.py` | Transformers `generate` check of Pythia final: precision x BOS prefix |

Our own FP32 samples use [`../decoding-presets/precision-fp32.json`](../decoding-presets/precision-fp32.json)
(same checkpoint, prompts, decoders, seeds and 128-token cap, plus one extra
T=0.85/top-p 0.95/penalty 1.1 decoder). Our BF16 samples are the final-checkpoint rows of the
earlier [repetition study](../repetition/README.md) (`previous-comparison` and `low-temperature`).

## Method

- Same ten prompts, 128 new tokens. Decoders: greedy; T=0.1/top-k 50 and T=0.8/top-k 50 with
  seeds 1000+i and 2000+i; T=1.0/top-p 0.95 with seed 1000+i.
- Primary comparison: full FP32 forward passes, TF32 disabled, tokenizer-default prefixes
  (Simple LLM uses its trained BOS). External models use Transformers eager attention; ours
  uses its original custom SDPA. A shared protocol, not bit-identical implementations.
- Tokenizer-independent metric: repeated normalized character 16-grams over the first 128
  normalized characters, on prompt/seed pairs eligible in every model (short outputs are
  excluded jointly). Word 4-grams over whole completions are reported as a secondary metric.
  Native-token 4-gram rates are not comparable across tokenizers. No metric measures correctness.
- The initial run used BF16 autocast; it is kept on HF as a precision-confounded negative result.
- Corpus audit: first 64 eligible training documents per source in content-hash order
  (768 total), first eight nonoverlapping 128-token windows per document, plus
  within-document and cross-document duplicate normalized lines (lines of 20+ characters).

## Key findings

Repeated character 16-grams, first 128 normalized characters, FP32:

| Model | Greedy | T=0.1 / top-k50 | T=0.8 / top-k50 | T=1.0 / top-p0.95 |
|---|---:|---:|---:|---:|
| Simple LLM 124M / 5B | 32.8% | 30.5% | 4.7% | 2.3% |
| Pythia 160M / 4.19B | 43.6% | 38.7% | 9.8% | 0.0% |
| Pythia 160M / 299.89B | 13.6% | 16.7% | 2.5% | 1.2% |
| SmolLM 135M / 600B | 12.6% | 11.2% | 1.4% | 1.7% |

(n = 10 greedy, 20 at T=0.1, 19 at T=0.8, 10 at T=1.0.)

- Our early looping is not an outlier relative to Pythia at a similar token budget; longer-trained
  Pythia and SmolLM repeat much less. The within-Pythia comparison points at training duration but
  does not isolate data quality, optimization or implementation issues in ours.
- Whole-completion greedy word 4-gram repetition: 71.8% ours, 66.7% early Pythia, 55.9% final
  Pythia, 36.9% SmolLM.
- **Precision is not the fix (negative result):** full FP32 gives 77.20% native-token greedy
  repetition vs 77.12% in BF16; 9/10 greedy completions are token-identical.
- **BF16 confound (negative result):** the initial BF16 external table made final Pythia look badly
  repetitive (66.8% greedy word4). A manual-loop vs Transformers `generate` check reproduced the
  BF16 outputs exactly, but FP32 changed them (e.g. Pythia answers "Paris" in FP32 and loops in
  BF16). Not a claim that BF16 universally breaks these models.
- **Corpus windows are far less repetitive than generations.** Token 4-gram repetition in 128-token
  windows / whole documents / duplicate lines within documents: educational web 1.9% / 4.8% / 0.0%,
  Python 7.9% / 25.0% / 10.1%, mathematics 7.1% / 18.6% / 3.5%, arXiv 3.6% / 31.0% / 1.1%,
  versus about 77% for our greedy generations. Descriptive, not causal.
- 229 normalized lines appear in more than one sampled document; the most frequent are table-wrapper
  extraction markers (29 and 19 documents) and a competing-interests declaration (14). Manual review
  of six high-repetition documents found navigation/browser boilerplate, quoted forum questions,
  unit-test scaffolding and license headers: candidates for targeted cleanup, not a prevalence estimate.
- Ten prompts were reused from earlier investigations; treat all of this as exploratory evidence.

## Suggested next experiments

1. Extraction cleanup (boilerplate/forum-quote removal plus near-deduplication) at a fixed model,
   tokenizer, mixture and token budget, preserving license/provenance records.
2. Current mixture vs a more prose-heavy educational mixture at the same budget, with frozen
   parent-level evaluation sets.
3. Small LR sweep (e.g. 3e-4 / 6e-4 / 1e-3) on the chosen corpus; test context curriculum separately.
4. Only then a staged 20–50B-token run with more unique data and a deliberately revised schedule.

## Rerun

From `training/`, with Modal credentials. Each launch is one bounded call with `retries=0` that
refuses work 24 hours after launch (see the toolkit README); outputs go to the ignored
`artifacts/` folder.

```sh
modal run --detach -m sampling.modal_sample --recipe analysis/repetition-baselines/external-fp32.json \
  --output artifacts/repetition-baselines/external-results.json
modal run --detach -m sampling.modal_sample --recipe analysis/decoding-presets/precision-fp32.json \
  --output artifacts/repetition-baselines/simple-llm-fp32.json
modal run analysis/repetition-baselines/modal_corpus_patterns.py   # CPU, 10-minute timeout
modal run analysis/repetition-baselines/modal_verify_pythia.py     # H100, 10-minute timeout
# optional: initial BF16 outputs
modal run --detach -m sampling.modal_sample --recipe analysis/repetition-baselines/external-bf16.json \
  --output artifacts/repetition-baselines/external-results-bf16.json
```

The corpus audit needs read access to the `lm-exploration-mixture-data` volume. Fresh toolkit
results name the merged seeds `topk50_t0.1` and `sampled_08` rather than the archived
per-seed mode names; the report groups by decoding settings, so tables are unaffected.

## Rebuild the report

Download `external-results.json`, `external-results-bf16.json` and `simple-llm-fp32.json` from
the HF folder above into `artifacts/repetition-baselines/`, and
`previous-comparison-20261018.json` and `low-temperature-20261018.json` from `analysis/repetition/`
in the model repository `jfrick/scholarly-code-modern-gpt-124m-step2` into
`artifacts/repetition/`. Then, from `training/`, select our final-checkpoint BF16 rows for the
shared decoders and build both reports:

```sh
jq '.checkpoints |= [last | .samples |= map(select(.mode | IN("greedy", "sampled_08_seed1", "sampled_08_seed2", "sampled_10_topp95")))]' \
  artifacts/repetition/previous-comparison-20261018.json > artifacts/repetition-baselines/simple-llm-bf16-previous.json
jq '.checkpoints |= [last | .samples |= map(select(.mode | startswith("topk50_t0.1_")))]' \
  artifacts/repetition/low-temperature-20261018.json > artifacts/repetition-baselines/simple-llm-bf16-low-temperature.json

uv run python -m sampling.report \
  artifacts/repetition-baselines/simple-llm-fp32.json artifacts/repetition-baselines/external-results.json \
  --reference artifacts/repetition-baselines/simple-llm-bf16-previous.json \
  --reference artifacts/repetition-baselines/simple-llm-bf16-low-temperature.json \
  --group-by decoding --paired --output-dir artifacts/repetition-baselines/report-fp32
uv run python -m sampling.report \
  artifacts/repetition-baselines/simple-llm-bf16-previous.json \
  artifacts/repetition-baselines/simple-llm-bf16-low-temperature.json \
  artifacts/repetition-baselines/external-results-bf16.json \
  --group-by decoding --paired --output-dir artifacts/repetition-baselines/report-bf16
```

Our model appears as run `step-64227`. In `summary.json`, `char_rep16_first128` and
`char_rep16_first256` (with `_n`) reproduce the archived `paired-prefix-summary*.json`, and N,
EOS, Token4, Word4 and Char16 reproduce the archived `metrics-table*.md`. The archived tables'
`First256char` column was unpaired; the new report pairs it. The FP32 report's `Identical` column
compares our FP32 samples with the BF16 ones (9/10 greedy). The extra T=0.85/penalty 1.1 row has
no external counterpart, so its paired prefix metrics are empty.
