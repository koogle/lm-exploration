# Simple LLM repetition: external baselines and corpus patterns

Bounded diagnostic comparing the final Simple LLM 124M / 5B checkpoint with base models
Pythia-160M (step 2000, about 4.19B tokens; final, 299.89B tokens) and SmolLM-135M (600B
tokens), plus a read-only audit of the training corpus. No training updates.

Raw outputs, all completions, metrics tables, the corpus audit and manual notes are on
Hugging Face:
[`jfrick/lm-exploration-results` / `pr-23/training/analysis/repetition-baselines/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-23/training/analysis/repetition-baselines)
(private dataset; archived in commit `56e1e8b`, see its `pr-23/README.md` for a file list).

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
- `modal_simple_llm_fp32.py` is derived from `training/analysis/decoding-presets/modal_precision.py`
  in [PR #20](https://github.com/koogle/lm-exploration/pull/20) (source SHA256
  `650f3e02…a9fa3`); only the CLI/output naming changed.

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

From this directory, with Modal credentials and read access to the project data volume. Outputs
go to the ignored `training/artifacts/repetition-baselines/` by default (each script accepts
`--output`).

```sh
modal run modal_baselines.py
modal run modal_simple_llm_fp32.py
modal run modal_corpus_patterns.py
modal run modal_verify_pythia.py
modal run modal_baselines_bf16.py   # optional: initial BF16 outputs
```

The GPU functions run on one H100 each with 30-minute (verification: 10-minute) timeouts and no
retries; the corpus audit is CPU-only with a ten-minute timeout and read-only SQLite access. In the
managed environment, run Modal through truststore:
`python -c "import truststore,runpy; truststore.inject_into_ssl(); runpy.run_module('modal',run_name='__main__')" run modal_baselines.py`.

`build_report.py` derives the summary, metrics table, paired-prefix summary and completions file.
It also needs `previous-comparison-20261018.json` and `low-temperature-20261018.json` from
`analysis/repetition/` in the model repository `jfrick/scholarly-code-modern-gpt-124m-step2`:

```sh
python build_report.py ../../artifacts/repetition-baselines --repetition ../../artifacts/repetition
python build_report.py ../../artifacts/repetition-baselines --repetition ../../artifacts/repetition --bf16
```

To rebuild from the archived raw outputs instead of rerunning Modal, download
`external-results*.json` and `simple-llm-fp32.json` from the HF folder above into the input folder.
Rebuilding from those files reproduces the archived derived outputs byte for byte.
