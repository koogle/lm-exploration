# Distillation validation ladder

A bounded experiment that answers, before any large student run, whether the
Qwen3.5-9B Python/math trajectories can teach a small model anything measurable,
and whether more data or a bigger model helps. **Status: campaigns
`distill-validation-20261023-01`, `-02-covering`, `-03-qwen-200m`,
`distill-validation-20261024-04-vllm-reeval` and `-05-vllm-native` ran on 2026-10-23/24;
findings below.**

Data: the private HF dataset `jfrick/qwen35-python-math-trajectories-20261022`
(native Qwen3.5 token IDs plus the teacher's top-64 log probabilities per
position). The run pins the dataset commit it reads and records it in its results.

## Terms

- **CE (cross-entropy) training:** the student learns only the token the teacher actually
  wrote at each position (a one-hot target), the ordinary next-token objective.
- **KD (knowledge distillation) training:** the student also learns the teacher's full
  next-token probability distribution, stored as its top-16 tokens plus one bucket for the
  remaining probability. The loss is `0.5 · CE + 0.5 · soft CE`, where soft CE is the
  cross-entropy against that distribution.
- **CE per byte (metric):** held-out cross-entropy in nats per UTF-8 byte of text, which
  can be compared across vocabularies; lower is better.
- **Pass@1:** the share of problems whose single greedy answer passes all hidden tests.

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
   after full fine-tuning on the first 50M and 200M positions of the same stream. If it
   does not improve, the data or the evaluation is the bottleneck, not student size.
   Its training code is in commit `883cffe`; its results are archived on HF.
3. **From-scratch ladder** (`plan.json`), all one pass over a nested prefix of one
   fixed chunk order (50M ⊂ 200M ⊂ 500M positions; no repeats):

   | Arm | Shape | Vocabulary | Backbone / total params | Budgets |
   | --- | --- | --- | --- | --- |
   | `trim-small` | 10 × 512 | 16,384 trimmed | 31.5M / 39.9M | 50M, 200M, 500M |
   | `trim-large` | 18 × 832 | 16,384 trimmed | 149.7M / 163.4M | 50M, 200M, 500M |
   | `full-small` | 10 × 512 | 248,320 native | 31.5M / 158.7M | 200M (500M did not fit; see findings) |

   Each cell is trained twice, with CE (`-ce` runs) and with KD (`-kd` runs); see Terms.
   `full-small` is the "150M total with Qwen's vocabulary" shape. Compared with
   `trim-small` it isolates the vocabulary: same backbone, so roughly 4× more compute
   goes into the output layer.

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
- **Inference.** `vllm_model.py` is the student decoder written for vLLM (same
  parameter names, interleaved RoPE, biased projections, tied embeddings); vLLM
  registers it in-process and loads the training checkpoint `model.pt` directly
  (`load_format="pt"`). `vllm_eval.py` generates from token IDs without a tokenizer and
  decodes through the student vocabulary; every evaluation records the prompt
  log-probability gap between vLLM (BF16) and the PyTorch checkpoint (FP32). The
  teacher also runs on vLLM.
- **Grading.** A CPU Modal function (`grade_job`) runs each case in its own interpreter
  with a timeout; a case passes only if every test runs and the process then prints a
  random marker.

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
# Cancel every recorded call of the campaign.
distillation/validation/.venv/bin/modal run distillation/validation/modal_app.py::stop
```

`VALIDATION_PLAN=<file>` selects another plan. Follow-up plans reuse an earlier
campaign's shards and evaluation sets (`data_campaign`) and may skip the teacher:
`plan-covering.json` retrains one arm with the covering vocabulary, and
`plan-reeval-vllm.json` re-evaluates saved students (`checkpoint`) without retraining.

**Runtime limits.** `campaign` has a 12-hour absolute deadline (`budget_seconds`,
launch + 43,200 s). Every child gets `retries=0` and a timeout of
`min(its own limit, remaining budget)`. Children: data prep 1 h, MBPP 15 min,
teacher 90 min, student 6 h, grading 1 h. The supervisor runs non-preemptible,
records all call IDs in the Modal Dict `lm-distillation-validation-calls`
and in `supervision.json` on the volume, refuses to start work with less than 5
minutes left, and cancels everything at the deadline or on a genuine failure. If the
supervisor itself is interrupted, children keep running within their own capped
timeouts and a restarted input adopts them. Finished
stages (`stages/*.json`) and models already in `results.json` are skipped on
relaunch, so a relaunch after a failure does not repeat data preparation.

**Cost (estimated, not reconciled with billing).** About 45–55 H100-hours across the
four campaigns, of which about 10 were lost to the preemptions and cancelled runs
described below.

**Archiving.** Each upload is a separate app (no retries, bounded) that verifies every
file's size on HF and does not touch running campaigns:

```sh
M=distillation/validation/modal_app.py
# Prepared shards of the plan's campaign (shards-card.md becomes the card).
distillation/validation/.venv/bin/modal run $M::archive --repo-id jfrick/qwen35-distillation-shards-20261023
# Results, grades, completions, metrics and supervision records (no weights).
distillation/validation/.venv/bin/modal run $M::archive_outputs --campaigns CAMPAIGN[,CAMPAIGN...]
# Final student weights (weights-card.md becomes the card).
distillation/validation/.venv/bin/modal run $M::archive_checkpoints --campaigns CAMPAIGN[,CAMPAIGN...]
```

Outputs go to the Modal volume `lm-exploration-distillation-validation` under
`/<campaign>/`: `manifest.json`, `stages/`, `eval-items.json`, `teacher/`,
`students/<arm>/` (metrics, summary, completions, `model.pt`, `vllm/`), `grades/`
and `results.json`. Findings below are written by hand.

## Tests

```sh
python -m unittest discover -s distillation/validation/tests -v
ruff check distillation && ruff format --check distillation
```

They need torch and numpy and are split per module (`tests/test_*.py`). They check
the soft loss against a hand computation, cross-entropy against PyTorch, packing and
stream order (including the covering vocabulary's piece expansion), covering
spelling, a short training run, the vLLM model directory (config, checkpoint link and
the weights as vLLM's `.pt` loader reads them), supervisor polling and adoption, the
MBPP conversion and the grader on fixed snippets. vLLM loading and parity are checked on the GPU in
every evaluation (see the re-evaluation below).

## Findings

Campaign `distill-validation-20261023-01`, dataset revision `5046680`: 547.7M training
tokens in 536,959 exercises, 11.0M held-out tokens. Greedy pass@1 with 95% Wilson
intervals; loss is held-out CE per UTF-8 byte. Full outputs: see [Archive](#archive).

| Model | Data | Held-out curriculum (137) | MBPP (427) | CE/byte |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B teacher (ceiling) | – | 83.2% [76.1, 88.5] | 76.8% [72.6, 80.6] | – |
| Qwen3.5-0.8B-Base, before | – | 10.2% [6.2, 16.4] | 12.9% [10.0, 16.4] | 0.305 |
| Qwen3.5-0.8B-Base, fine-tuned | 50M | 62.0% [53.7, 69.7] | 41.7% [37.1, 46.4] | 0.171 |
| Qwen3.5-0.8B-Base, fine-tuned CE ¹ | 200M | 78.8% [71.3, 84.8] | 42.9% [38.2, 47.6] | 0.153 |
| Qwen3.5-0.8B-Base, fine-tuned KD ¹ | 200M | 81.0% [73.6, 86.7] | 44.0% [39.4, 48.8] | 0.151 |
| trim-small CE | 50M | 8.0% [4.5, 13.8] | 0.2% [0.0, 1.3] | 0.493 |
| trim-small CE | 200M | 35.0% [27.6, 43.3] | 4.0% [2.5, 6.3] | 0.291 |
| trim-small CE | 500M | 49.6% [41.4, 57.9] | 9.1% [6.8, 12.2] | 0.241 |
| trim-small KD | 50M | 5.8% [3.0, 11.1] | 0.7% [0.2, 2.0] | 0.491 |
| trim-small KD | 200M | 32.8% [25.5, 41.1] | 4.2% [2.7, 6.6] | 0.287 |
| trim-small KD | 500M | 44.5% [36.5, 52.9] | 5.9% [4.0, 8.5] | 0.235 |
| trim-large CE | 50M | 9.5% [5.6, 15.6] | 1.9% [1.0, 3.7] | 0.423 |
| trim-large CE | 200M | 44.5% [36.5, 52.9] | 5.2% [3.4, 7.7] | 0.258 |
| trim-large CE | 500M | 54.0% [45.7, 62.1] | 11.0% [8.4, 14.3] | 0.212 |
| trim-large KD | 50M | 10.9% [6.7, 17.3] | 0.9% [0.4, 2.4] | 0.413 |
| trim-large KD | 200M | 44.5% [36.5, 52.9] | 5.9% [4.0, 8.5] | 0.250 |
| trim-large KD | 500M | 62.8% [54.4, 70.4] | 13.6% [10.7, 17.2] | 0.204 |

Vocabulary at equal backbone (31.5M, 10 × 512) and 200M tokens:

| Vocabulary | Objective | Held-out curriculum | MBPP | CE/byte | Train time |
| --- | --- | ---: | ---: | ---: | ---: |
| 16,384 trimmed | CE | 35.0% [27.6, 43.3] | 4.0% [2.5, 6.3] | 0.291 | 20.5 min |
| 248,320 native | CE | 33.6% [26.2, 41.8] | 4.0% [2.5, 6.3] | 0.295 | 152.3 min |
| 16,384 trimmed | KD | 32.8% [25.5, 41.1] | 4.2% [2.7, 6.6] | 0.287 | 23.2 min |
| 248,320 native | KD | 31.4% [24.2, 39.6] | 4.4% [2.9, 6.8] | 0.289 | 151.8 min |

¹ Campaign `distill-validation-20261023-03-qwen-200m`, after the control was sped up
from about 7K to 19–23K tokens/s (`fla-core` kernels and a soft loss without dense
vocabulary temporaries; H100 probe reports in `distillation-validation/control-probes/`).

Student scores above come from the original PyTorch decoder. Re-evaluating all 15
saved students through the Llama export and vLLM (`distill-validation-20261024-04-vllm-reeval`)
reproduces them within noise: per model, held-out passes changed by −4 to +4 of 137
(mean +0.3) and MBPP passes by −2 to +2 of 427 (mean +0.3). The vLLM-vs-PyTorch prompt
log-probability gap was 0.011–0.014 on average and at most 0.06–0.10 per checkpoint.

The current path runs the student itself in vLLM (`vllm_model.py`, loading `model.pt`
directly). Its re-evaluation of the same 15 checkpoints
(`distill-validation-20261024-05-vllm-native`) also reproduces them: held-out passes
changed by −3 to +4 (mean +0.5) and MBPP passes by −4 to +3 (mean +0.1); the prompt
log-probability gap was 0.012–0.016 on average and at most 0.06–0.14. The table shows
the export-based run (`-04`):

| Student | Held-out, original → vLLM | MBPP, original → vLLM |
| --- | ---: | ---: |
| trim-small-50m-ce / -kd | 11 → 10 / 8 → 8 | 1 → 1 / 3 → 3 |
| trim-small-200m-ce / -kd | 48 → 50 / 45 → 46 | 17 → 19 / 18 → 18 |
| trim-small-500m-ce / -kd | 68 → 68 / 61 → 65 | 39 → 41 / 25 → 24 |
| trim-large-50m-ce / -kd | 13 → 13 / 15 → 15 | 8 → 8 / 4 → 4 |
| trim-large-200m-ce / -kd | 61 → 62 / 61 → 62 | 22 → 22 / 25 → 26 |
| trim-large-500m-ce / -kd | 74 → 76 / 86 → 84 | 47 → 49 / 58 → 59 |
| full-small-200m-ce / -kd | 46 → 48 / 43 → 41 | 17 → 17 / 19 → 17 |
| cover-large-500m-kd | 83 → 79 | 48 → 48 |

- **The data teaches the task.** Fine-tuning the pretrained 0.8B control on 50M tokens
  (about a tenth of the data) raised MBPP from 12.9% to 41.7% and the held-out set from
  10.2% to 62.0%. Part of the base score is format (297/427 base answers parsed, 406
  after), but most of the gain is correctness on human-written problems outside the
  teacher's templates.
- **From scratch, more data keeps helping; there is no plateau at 500M.** The 31.5M
  backbone went from 8% to 35% to 50% held-out and from 0.2% to 4% to 9% MBPP at 50M,
  200M and 500M tokens. These are the first non-zero results of any distillation
  attempt in this repository (the earlier best was 1/8 diagnostic tasks).
- **Size helps at equal data.** At 500M tokens the 150M backbone beats the 31.5M one
  on loss (0.204–0.212 vs 0.235–0.241) and on both sets, although the pass-rate
  intervals overlap.
- **KD always lowers loss; its effect on pass rate depends on size.** On the small
  model KD was neutral or worse (500M MBPP 5.9% vs 9.1%); on the large model at 500M it
  was the best student (62.8% held-out, 13.6% MBPP vs 54.0% / 11.0% for CE). Intervals
  overlap, so this is a lead, not a result. KD used α = 0.5 at temperature 1 without
  tuning.
- **The native vocabulary buys nothing.** With Qwen's tokenizer shared by every arm,
  the 248K vocabulary matched the trimmed one on both sets and was marginally worse on
  loss, at 7.4× the training time (partly the dense soft-loss code since replaced;
  the output layer alone is about 4× the backbone's compute at this width).
- **Removing UNK from MBPP prompts did not help.** Every MBPP prompt contained tokens
  outside the trimmed vocabulary (`assert` in all 427, ` python` in 126; 3.4 UNK per
  prompt). Retraining `trim-large-500m-kd` with the covering vocabulary
  (`distill-validation-20261023-02-covering`, no UNK, exact round-trips) gave 60.6%
  [52.2, 68.4] held-out and 11.2% [8.6, 14.6] MBPP (413/427 parsed, CE/byte 0.206),
  within noise of the trimmed run. The words are now readable as pieces, but the
  training data never uses them, so the gap is knowledge rather than vocabulary.
- **More data closes the in-distribution gap for the pretrained model, not MBPP.**
  From 50M to 200M tokens the 0.8B rose from 62.0% to 78.8–81.0% held-out, close to the
  teacher's 83.2%, while MBPP moved only from 41.7% to 42.9–44.0%. KD was within noise
  of CE (slightly ahead on both sets and on loss).
- **The from-scratch gap is general knowledge, not data.** The best student reaches
  63% held-out but only 14% MBPP, against 42% for the 0.8B model fine-tuned on a tenth
  of the data. Narrow exercises alone do not supply the Python knowledge MBPP needs.
- **Teacher test agreement is low, and mostly a test problem.** Of 548 held-out
  exercises with at least three teacher-written tests, the original solution passed
  all of them in 137 (25%); 380 failures were assertion mismatches. In the cases read
  by hand the code was right and the expected values (computed by the teacher without
  thinking) were wrong. Treat 25% as a loose lower bound on data correctness.
- **Held-out overlap is unresolved.** Held-out problem statements share at most 25% of
  their word 3-grams with a 1,654-record training sample; one solution-code
  near-duplicate (5-gram Jaccard ≥ 0.5) in that sample extrapolates to about two per
  item across the full training set, with a very wide interval. MBPP is the clean
  generalization measure.

Operational notes, negative ones included:

- The first launch counted unfinished calls as failed (`get(timeout=0)` raises the
  builtin `TimeoutError`). The second hit HF HTTP 429 at 512 concurrent downloads.
  Prep now reuses finished shards and runs 16 containers × 2 downloads.
- The workspace's 10-GPU limit was held by the teacher-generation app; the GPU stages
  waited about 1.5 hours until it was reduced to 5.
- The 248K-vocabulary students trained at about 21.8K positions/s against about 170K
  for the trimmed small model (about 8× slower), so the 500M runs could not finish
  within the 6-hour job limit. They were cancelled at about 230M positions and
  replaced by 200M runs.
- The 0.8B control fine-tune first ran at about 7K tokens/s: `flash-linear-attention`
  0.5 keeps its kernels in `fla-core`, which was not installed, so Transformers
  silently used PyTorch reference loops; the dense soft loss was the next bottleneck.
- Modal preempted two supervisors. Their shutdown went through the failure path and
  cancelled children (the full-vocabulary runs at about 1.3 hours, the 0.8B 200M runs at
  77%); the supervisor is now non-preemptible and adopts its children on restart.
  Training runs do not checkpoint mid-run, so an interruption restarts them.

## Archive

- Training shards: private HF dataset `jfrick/qwen35-distillation-shards-20261023`.
- Results, grades, completions, metrics and supervision records of all campaigns (`-01`,
  `-02-covering`, `-03-qwen-200m`, `-04-vllm-reeval`, `-05-vllm-native`),
  plus the control throughput probes: HF dataset `jfrick/lm-exploration-results`,
  folder `distillation-validation/`.
- Final weights (`students/*/model.pt`, the earlier Llama exports in `students/*/export/`
  from campaign `-04-vllm-reeval`, and the
  fine-tuned 0.8B models in `control/*/model/`): private HF dataset
  `jfrick/distillation-validation-checkpoints-20261023`.
