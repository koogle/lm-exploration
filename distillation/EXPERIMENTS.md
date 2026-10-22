# Distillation experiments (2026-10-21/22)

Recorded runs of the [distillation package](README.md). Generated reports, result JSONs,
samples and the original run docs are archived in the private dataset
[`jfrick/lm-exploration-results`, folder `pr-31/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-31)
(results under `pr-31/distillation/results/`, former run docs under `pr-31/distillation/`).
Run folders with checkpoints, metrics, source snapshots and raw rollouts stay on the
Modal volume `lm-exploration-distillation`.

Each recorded job has a recipe in [`recipes/`](recipes/) naming its teacher revision,
parent, data and run ID. Those runs were launched with the per-run launchers of PR #31
at commit `6d39610` or earlier, which the generic launchers replaced; `provenance` in each
recipe names the old launcher and the archived results. Byte and shared checkpoints pin
their exact source hashes, so resuming them needs the source archived in the run folder.

Common inputs: parent `simple-llm-code60-unique-10b-20261020-r01-s04/step-000153342.pt`
(124,389,632 parameters, about 20.879B lifetime tokens, frozen 32K tokenizer
`add5b2c7542e4efded329735825ecdcb94df1432617c3d3e4b7508aa459432f9`); code snapshot `scholarly-code/readiness-20261020-v2` (multilingual
code-only database, splits and held-out exclusions preserved). KL teacher
`Qwen/Qwen3-4B-Base@906bfd4b`, chosen as a modest pretrained base, not as the best
available model (candidate list: `results/teacher-candidates-20261021.json`). Generation
teacher `Qwen/Qwen3-8B@b968826d`, thinking disabled.

## Byte-marginal pilot: negative

Recipe [`byte-pilot`](recipes/byte-pilot.json); report `results/pilot-20261021.md`.

- 261,907 student loss tokens, 1,133 updates on one L40S at 1,711 tokens/s; 224,675
  aligned boundaries (85.8% of loss tokens) received byte-marginal KL; 16 windows skipped
  for Unicode/normalization. Peak allocated memory 10.6 GB.
- Code CE on 16 fixed windows (3,724 tokens) **worsened 1.2724 -> 1.3782**.
- Larger holdout ([`byte-pilot-holdout`](recipes/byte-pilot-holdout.json); up to 512 documents,
  1,578 windows, 342,549 tokens): parent 1.7349, steps 500 / 1,000 / 1,133 at
  1.8245 / 1.8496 / 1.8625. CE rose monotonically with training.
- The first attempt failed before optimization because Qwen's tokenizer declares Unicode
  normalization; the recorded run skips windows that normalization would change.
- No CE-only control was run, so the regression is not attributed to the KL term alone.
  This recipe did not earn a scale-up.

## vLLM export parity: passed

`modal_vllm::parity` on byte-pilot step 1,000 (L4, FP32, vLLM 0.11.0): 5/5 greedy tokens
matched, maximum log-probability error 1.16e-5, maximum export logit error 2.55e-5
(`results/vllm-parity-20261021.json`). Serving parity only; harness parity and BF16 were
not qualified.

## Shared-tokenizer KL, 50M tokens

Recipes [`shared-smoke`](recipes/shared-smoke.json) and
[`shared-50m`](recipes/shared-50m.json). Switching to Qwen's 151,936-token vocabulary grows
the student to 215,910,656 parameters (tied embeddings).

- Smoke: 16,384 tokens, 3 updates, CE 3.5328 -> 3.5249.
- 50M: 50,000,000 loss-bearing tokens, 7,162 updates, about 14.3K tokens/s on one H100
  (about one hour). Validation CE (128 documents, 214 windows, 75,394 tokens)
  **3.533 -> 2.275**; final training token KL 1.228. No validation-based stopping.
- Raw CE is in the Qwen tokenizer and is not comparable with the byte pilot's 32K CE.
  No matched CE-only control, so the gain cannot be attributed to teacher supervision
  rather than continued training after the vocabulary change.
- Greedy continuations ([`shared-50m-generations`](recipes/shared-50m-generations.json);
  steps 0 to 7,162 on three prompts): the addition return becomes correct; factorial and
  order-preserving deduplication stay incorrect, and many continuations repeat
  definitions. Generated code was not executed; qualitative only.

## Qwen3-8B curriculum generation and students

Recipes [`curriculum-01`](recipes/curriculum-01.json),
[`curriculum-02`](recipes/curriculum-02.json) and
[`curriculum-students`](recipes/curriculum-students.json); report
`results/curriculum-generation-20261021.md`.

| Run | Accepted / 512 | Train tokens | Validation tokens | Generation |
| --- | ---: | ---: | ---: | ---: |
| `qwen8b-python-curriculum-20261021-01` | 456 (89.1%) | 66,732 | 21,754 | 90.0 s |
| `qwen8b-python-curriculum-20261021-02` | 500 (97.7%) | 74,672 | 25,516 | 91.8 s |

- Run 01 accepted only 20/64 factorial tasks (the teacher computed `n!` for `(n+1)!`).
  Run 02 rewrote that specification; all 64 passed. The batches share families and
  constants and are not independent datasets. Explanations are unverified teacher prose.
- Fresh 26,776,960- and 48,308,408-parameter students, one pass over run 02's 74,672
  train tokens (24 updates each): validation CE 10.466 -> 8.096 -> 7.397 and
  10.523 -> 7.310 -> 6.399 (initial, midpoint, final). **0/8 diagnostic tasks at every
  checkpoint.** This qualifies the data, training and evaluation plumbing only.

## 500M-token sweep: stopped

Run `python-distillation-500m-20261021-01` (`modal_scale.py`, `scale_train.py`;
reports `results/scale-500m-20261021.md` and the `python-distillation-500m-*` JSONs).
Four Qwen3-8B producers generated a broader 62-topic Python/math corpus (syntax-checked,
not execution-verified) toward 500M original-tokenizer tokens; two fresh students
targeted 500M tokens each and the 215.9M shared model a 4.5M update. Sequence
distillation with all-token CE.

- 215.9M from the shared-50M checkpoint: exactly 4,500,000 tokens, 817 updates;
  validation CE 2.569 -> 1.621 (25,240 Qwen tokens); diagnostics 0/8 -> 1/8.
- Fresh 26.8M / 48.3M after their first 5,232,663-token segment (476 updates): CE
  10.466 -> 2.853 and 10.523 -> 2.852 (25,516 original tokens).
- The user stopped the sweep on 2026-10-21. Durable progress: 27M student 135,406,075
  tokens, 48M student 85,406,075. The 500M targets were not reached.
- `modal_archive_sweep.py` archived all producer data (243,440,328 original-tokenizer
  targets, 665,534 examples) and the latest student checkpoints to the private dataset
  `jfrick/python-math-500m-sweep-stopped-20261021`; its `index.json` lists per-file
  revisions and SHA256 hashes. Older weight snapshots remain on the volume.

## 215.9M shared-Qwen continuation to 4.5B tokens: paused

In progress; plan and operations are in [BILLION_RUN.md](BILLION_RUN.md). Recorded so far:
an orchestration interruption at 21,639,566 targets (validation CE 1.55013, versus
1.62070 at 4.5M), a user correction to a single pass over fresh data before any replay,
a frozen 103,669,090-token initial prefix with resume from the 61.65M-token checkpoint,
and the first archive commit of 11,396,274 targets / 31,006 examples to
`jfrick/python-math-qwen3-distillation-20261021`. The eight-GPU student was still queued
at the last check, so no eight-GPU throughput is claimed. On 2026-10-22 the Qwen3-8B
teacher campaign was paused to qualify Qwen3.5 generation
([PR #39](https://github.com/koogle/lm-exploration/pull/39)).

## Scaling notes for follow-ups

`python -m distillation.scaling --accepted-tokens N` reproduces the parameter table: with
the 32K vocabulary, 26.78M / 48.31M / 124.39M total parameters; with Qwen's vocabulary,
72.54M / 109.32M / 215.91M, of which embeddings are 58.34M / 77.79M / 116.69M. The 20
tokens/parameter heuristic is a from-scratch compute-allocation starting point, not a
capacity limit or a continued-training budget. Before fitting any curve: verify
explanations, broaden templates, hold out whole templates, compare a matched CE-only
control and code-only versus code-plus-explanation data at equal exposure, and report
executable pass@1 on a frozen test set. Never compare raw CE across tokenizers.
