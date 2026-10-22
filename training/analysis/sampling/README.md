# Sampling studies toolkit

One launcher, one report and one set of repetition metrics for fixed-prompt sampling
studies. Each study folder holds only a recipe (JSON) and a README of findings;
generated outputs belong on Hugging Face. uv installs this folder as `sampling`.

| File | Role |
| --- | --- |
| `recipe.py` | Recipe loading and validation (pinned commits, bounded generation) |
| `sampler.py` | Decoding loop shared by the launcher and the CPU tests |
| `metrics.py` | Repetition metrics; also used by `validate_repetition_data.py` |
| `modal_sample.py` | Modal launcher: one H100 call per recipe |
| `report.py` | Summary, metrics table and all-completions dump from result JSONs |
| `recipes/` | Recipes without a study folder |

## Recipe

```json
{
  "name": "decoding-presets",
  "precision": "bf16-autocast",
  "max_new_tokens": 128,
  "prompts": [{"text": "def add(a, b):\n    ", "group": "python"}, "Once upon a time,"],
  "modes": [{"name": "t085_p95_rp110", "temperature": 0.85, "top_p": 0.95,
             "repetition_penalty": 1.1, "seed_offsets": [1000, 2000]}],
  "models": [{"kind": "checkpoint", "name": "simple-llm-124m", "repo": "...",
              "source_revision": "<commit>", "model_source_sha256": "...",
              "tokenizer_sha256": "...", "parameters": 124389632,
              "prefix_token_ids": [1], "eos_token_id": 2,
              "checkpoints": [{"step": 64227, "revision": "<commit>"}]}]
}
```

- `precision`: `bf16-autocast` (FP32 weights, BF16 autocast, PyTorch TF32 defaults) or
  `fp32` (FP32 forward, CUDA matmul/cuDNN TF32 off, highest matmul precision).
- Each sample is seeded with `torch.manual_seed(seed_offset + prompt_index)`.
  Temperature 0 is greedy. Otherwise the order is: sign-aware repetition penalty (once per
  token identity in the prompt and generated prefix), temperature, optional top-k, then a
  nucleus that keeps the token crossing `top_p`. Generation stops at EOS or `max_new_tokens`.
- `checkpoint` models are our own published checkpoints. Model code comes only from
  `source_revision`. Its source and tokenizer hashes must match the recipe and provenance.
  Each checkpoint is checked for provenance step and source, weight SHA256, finite parameters
  and parameter count.
- `transformers` models (`repo`, `revision`, optional `parameters`, `trained_tokens`,
  `prefix_token_ids`) load in FP32 with eager attention and `trust_remote_code=False`. The
  revision must resolve to itself.
- All revisions must be full commit hashes. One recipe uses one model kind.

## Run

From `training/`, with Modal credentials and the `lm-exploration-training` volume:

```sh
modal run --detach -m sampling.modal_sample --recipe analysis/decoding-presets/decoding-presets.json \
  --output artifacts/decoding-presets/decoding-presets.json
```

Each launch makes one function call: one H100, 30-minute hard timeout, `retries=0`,
`max_containers=1`. The call refuses further samples 24 hours after launch, including
time spent queued. After each model it writes partial results to
`/runs/sampling/<name>.json` on the volume. Rerunning an identical recipe skips models
that are already complete. The output defaults to the ignored
`artifacts/sampling/<name>.json`.

In an environment whose outbound proxy uses the system CA store, launch through truststore
(TLS verification stays on):
`python -c 'import truststore; truststore.inject_into_ssl(); from modal.__main__ import main; main()' run --detach -m sampling.modal_sample --recipe ...`.

The result JSON has `recipe`, `precision`, TF32 flags, `torch_version`, `device`,
`elapsed_seconds` and `models[]`. Each model lists its provenance and `samples[]`, with
`prompt`, `prompt_group`, `mode`, decoding settings, `seed`, `generation_ids`,
`generated_tokens`, `ended_with_eos` and `completion`.

## Report

```sh
uv run python -m sampling.report RESULT.json [MORE.json ...] --output-dir DIR \
  [--reference EARLIER.json ...] [--group-by mode] [--group-by decoding] [--paired]
```

The report writes `summary.json`, `metrics-table.md` and `completions.md` to
`--output-dir`. It reads only the launcher format above. Archived results in the earlier
per-study format (`checkpoints[]`, with settings in top-level `modes`) need the report as
of revision `1e06fb8`.

- `--group-by` takes comma-separated sample fields, such as `mode`, `temperature`,
  `prompt_group` or `seed`. `decoding` stands for temperature, top-k, top-p and penalty.
  Each `--group-by` produces one table. Rows are always split by run.
- `--reference` matches samples by revision, prompt, seed and decoding settings, then
  counts token-identical completions and records the first divergent position.
- `--paired` averages the prefix metrics only over prompt/seed pairs eligible in every run.

Per-sample metrics (`metrics.py`):

- `token_rep4`: `1 - unique / all` native-token 4-grams of the generated tokens.
- `word_rep4` and `char_rep16`: computed on normalized text (lowercased, whitespace
  collapsed).
- `char_rep16_first128`, `char_rep16_first256` and `word_rep4_first32`: computed on a
  prefix window. They are `None` when the completion is shorter than the window.

Native-token rates are not comparable across tokenizers. No metric measures correctness.
