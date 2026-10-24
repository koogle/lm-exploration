# Distillation

Qwen-teacher distillation into the 124M Simple LLM was tried in
[PR #31](https://github.com/koogle/lm-exploration/pull/31) and
[PR #43](https://github.com/koogle/lm-exploration/pull/43) and closed without merging.
Only the reusable teacher-dataset modules remain here: `scale_data.py` (streaming
generation shards), `dataset_pipeline.py` (append-only publication) and
`dataset_archive.py` (HF dataset archiving), with `tests/test_dataset_pipeline.py`.

## Red flag: teacher and student tokenizers did not match

The student used its own frozen 32K tokenizer; the teachers (`Qwen/Qwen3-4B-Base` for KL,
`Qwen/Qwen3-8B` for rollouts) use Qwen's 151,936-token tokenizer. Every approach had to
work around that mismatch, and none produced a better model:

- **Byte-marginal KL** aligned teacher and student at shared byte boundaries. Only 85.8%
  of loss tokens received KL, Qwen's declared Unicode normalization broke the first
  attempt, and code CE **worsened** 1.2724 → 1.3782 (held-out 1.7349 → 1.8625).
- **Switching the student to Qwen's vocabulary** grew it from 124M to 216M parameters
  and reset comparability: CE improved 3.533 → 2.275 in Qwen tokens, but without a
  CE-only control the gain cannot be credited to the teacher.
- **Sequence distillation on Qwen3-8B rollouts** then tracked students in two incompatible
  units (32K-tokenizer and Qwen tokens), so CE could not be compared across them. The
  500M-token sweep was stopped short with at most 1/8 diagnostic tasks solved, and the
  planned Qwen3.5 teacher brings yet another native tokenizer.

Lesson: standardise on one teacher model and its tokenizer before distilling, and train
the student on that tokenizer from the start, or at least pick a teacher whose tokenizer
the student already shares. Do not handle cross-tokenizer KL divergence; treat any need
for alignment code as a reason to stop and redesign.

## Validation ladder

[`validation/`](validation/README.md) tests whether the Qwen3.5-9B trajectories teach
anything measurable before a large run: an execution-graded evaluation with the teacher
ceiling, a fine-tuned Qwen3.5-0.8B data-quality control, and a from-scratch ladder over
model size, token budget, CE versus KD, and a trimmed versus native vocabulary.

## Archive

Results, generated reports, run docs and the hand-written findings (`pr-43/EXPERIMENTS.md`,
`pr-43/BILLION_RUN.md`) are in the private dataset
[`jfrick/lm-exploration-results`, folder `pr-31/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-31).
Generated datasets: `jfrick/python-math-qwen3-distillation-20261021` and
`jfrick/python-math-500m-sweep-stopped-20261021`. Code is in the closed PRs' branch history.
