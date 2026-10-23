# Distillation validation ladder

A bounded experiment that answers, before any large student run, whether the
Qwen3.5-9B Python/math trajectories can teach a small model anything measurable,
and whether more data or a bigger model helps. **Status: implemented and
CPU-tested; not yet launched.** No results exist yet.

Data: the private HF dataset `jfrick/qwen35-python-math-trajectories-20261022`
(native Qwen3.5 token IDs plus the teacher's top-64 log probabilities per
position). The run pins the dataset commit it reads and records it in its results.

## Questions and arms

1. **Evaluation and teacher ceiling.** Execution-graded pass@1 (greedy) on two sets:
   - `mbpp`: all 427 sanitized MBPP problems (human-written tests), phrased in the
     training data's JSON format with the signature and first test shown.
   - `curriculum`: up to 400 exercises from held-out teacher chunks. The teacher
     writes 6–8 asserts from the problem text; an item is kept only when the
     teacher's original solution passes all of them. The kept fraction
     (`curriculum_agreement`) estimates how self-consistent the training data is.

   The teacher's own pass rate (chat template, thinking off) is the ceiling.
2. **Data-quality control.** Qwen3.5-0.8B-Base (same tokenizer) evaluated before and
   after full fine-tuning on the first 50M positions of the same stream. If it does
   not improve, the data or the evaluation is the bottleneck, not student size.
3. **From-scratch ladder** (`plan.json`), all one pass over a nested prefix of one
   fixed chunk order (50M ⊂ 200M ⊂ 500M positions; no repeats):

   | Arm | Shape | Vocabulary | Backbone / total params | Budgets |
   | --- | --- | --- | --- | --- |
   | `trim-small` | 10 × 512 | 16,384 trimmed | 31.5M / 39.9M | 50M, 200M, 500M |
   | `trim-large` | 18 × 832 | 16,384 trimmed | 149.7M / 163.4M | 50M, 200M, 500M |
   | `full-small` | 10 × 512 | 248,320 native | 31.5M / 158.7M | 500M |

   Each cell is trained twice: plain cross-entropy (`ce`) and `0.5 · CE + 0.5 ·
   soft CE` against the teacher (`kd`). `full-small` is the "150M total with Qwen's
   vocabulary" shape. Compared with `trim-small` it isolates the vocabulary: same
   backbone, so roughly 4× more compute goes into the output layer.

## Method

- **Split.** Chunks whose salted path hash falls in the lowest 2% are held out
  (validation loss and curriculum problems); the rest are trained on in a salted
  hash order. Only `accepted` records that ended with the stop token are used; the
  generation prompt is dropped, so students learn the exercise text itself.
- **Trimmed vocabulary.** PAD, BOS and UNK plus the 16,381 most frequent native IDs
  in the training split. Unseen IDs map to UNK. Coverage is recorded in the manifest.
- **Teacher signal.** Top-16 entries per position (99.7% of the mass in a sample)
  plus one residual bucket. The soft loss is the exact cross-entropy over the
  partition {top-16, everything else}; IDs merged into UNK are summed.
- **Students.** The repository's modern decoder (`training/10m/model.py`: RMSNorm,
  RoPE, SwiGLU, tied embeddings), 2,048-token packed windows with a BOS per
  document, 32 windows (65,536 positions) per update, AdamW (0.9, 0.95), weight
  decay 0.1, 2% warmup, cosine to 10%. Learning rates (small 1.5e-3, large 8e-4) are
  untuned single choices.
- **Metrics.** Held-out CE per token (only comparable within one vocabulary), CE per
  UTF-8 byte (comparable across vocabularies; trimmed UNK targets make the trimmed
  arms slightly optimistic), KL to the teacher, and pass@1 with 95% Wilson intervals.
- **Grading.** Every completion runs in a network-blocked Modal Sandbox. Each case
  gets its own interpreter with CPU, memory, file-size and process limits and must
  print a random marker. Model output is never executed outside the sandbox.

## Run

From the repository root, with Modal credentials and the existing Modal secret
`distillation-huggingface` (an `HF_TOKEN` that can read the private dataset):

```sh
uv venv --python 3.12 distillation/validation/.venv
uv pip install --python distillation/validation/.venv -r distillation/validation/vllm-overlay.lock
# Dry run: prints the plan and the effective timeouts; launches nothing.
distillation/validation/.venv/bin/modal run distillation/validation/modal_app.py::main
# Launch; the remote supervisor keeps running after the client disconnects.
distillation/validation/.venv/bin/modal run --detach distillation/validation/modal_app.py::main --launch
# Cancel every recorded call and sandbox of the campaign.
distillation/validation/.venv/bin/modal run distillation/validation/modal_app.py::stop
```

**Runtime limits.** `campaign` has a 12-hour absolute deadline (`budget_seconds`,
launch + 43,200 s). Every child gets `retries=0` and a timeout of
`min(its own limit, remaining budget)`. Children: data prep 1 h, MBPP 15 min,
teacher 90 min, control 4 h, student 6 h, grading sandbox 1 h. The supervisor
records all call and sandbox IDs in the Modal Dict `lm-distillation-validation-calls`
and in `supervision.json` on the volume, refuses to start work with less than 5
minutes left, and cancels everything at the deadline or on failure. Finished
stages (`stages/*.json`) and models already in `results.json` are skipped on
relaunch, so a relaunch after a failure does not repeat data preparation.

**Estimated cost (not measured).** About 64 CPU containers for data preparation
(~200 GB download), then 16 H100 jobs: roughly 10–15 H100-hours in total (the
three 500M-position 150M-scale students dominate), about $40–60 at the public
$3.95/H100-hour rate. Wall time depends on how many H100s Modal allocates; earlier
campaigns reached about ten at once.

**Training shards on HF.** After the data stage completes, upload the prepared
shards (about 50 GB) to a private dataset and verify every file's size:

```sh
distillation/validation/.venv/bin/modal run distillation/validation/modal_app.py::archive \
  --repo-id jfrick/qwen35-distillation-shards-20261023
```

This is a separate app, bounded at 6 hours with no retries, and it does not touch the
running campaign. The receipt (`shards-archive.json` on the volume) records the HF
revision. `shards-card.md` becomes the dataset card.

Outputs go to the Modal volume `lm-exploration-distillation-validation` under
`/<campaign>/`: `manifest.json`, `stages/`, `eval-items.json`, `teacher/`,
`control/`, `students/<arm>/` (metrics, summary, completions, `model.pt`),
`grades/` and `results.json`. Archive them to the HF dataset
`jfrick/lm-exploration-results` under `distillation-validation/<campaign>/` and
record the findings below by hand.

## Tests

```sh
python -m unittest discover -s distillation/validation/tests -v
ruff check distillation && ruff format --check distillation
```

They need torch and numpy. They check the
soft loss against a hand computation, packing and stream order, KV-cache decoding
against the full forward pass with left padding, a short training run, the MBPP
conversion and the grader on fixed snippets. A CPU smoke test on real dataset
chunks (prepare → trimmed and full vocabulary → train → validate → generate)
also passed. GPU throughput, vLLM loading of the fine-tuned control and the
sandbox were not exercised before launch.

## Findings

None yet: the campaign has not been launched.
