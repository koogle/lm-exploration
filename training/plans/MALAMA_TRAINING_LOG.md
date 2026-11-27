# Malama training log

A dated record of everything that was run to produce Malama v0.0.1, from pretraining to
the release: each launch, what it measured, what went wrong and the decision that
followed. Times are UTC and LiveCodeBench scores are v6 pass@1 in the thinking protocol
unless stated. The [run log](#run-log) lists every launch in one table; the
[day-by-day account](#day-by-day) below it tells what happened. Results are summarized in
[MALAMA_V0_0_1.md](MALAMA_V0_0_1.md) and
[CODING_REASONER_1B_FINDINGS.md](CODING_REASONER_1B_FINDINGS.md).

Days up to 21 November are written from the run records, the base model's card and the
findings documents. Days from 22 November are written from the reports of the session
that ran them.

## Run log

Every launch, by workflow run ID (dates UTC). Launches up to reasoning SFT v4 are from each
launch's workflow record and are discussed in
[CODING_REASONER_1B_FINDINGS.md](CODING_REASONER_1B_FINDINGS.md); their records are under
`training/results/coding-reasoner-1b/runs/<run ID>/` in `jfrick/lm-exploration-results`.
Records of later launches are under `pr-197/runs/` of the same dataset. LiveCodeBench scores are v6 pass@1 in the thinking protocol.

| Run ID | Workflow | Dates | Outcome |
|---|---|---|---|
| `kda-960m-foundation-20261110-01` | `kda-960m-foundation` | Nov 10–14 | Completed in seven launches: Malama Base, 100B tokens at 4K context ([card](https://huggingface.co/jfrick/malama-base)) |
| `kda-960m-fim-midtrain-20261114-01` | `kda-960m-fim-midtrain` | Nov 14–15 | Completed: about 15B tokens with fill-in-the-middle from Malama Base at 84.8B tokens |
| `kda-960m-lcb-format-sft-20261115-01` | `kda-960m-lcb-format-sft` | Nov 15 | Failed before training: the recipe was not prepared on the data volume |
| `kda-960m-lcb-format-sft-20261115-02` | `kda-960m-lcb-format-sft` | Nov 15 | Failed at startup: the trainer could not read the recipe's data files |
| `kda-960m-lcb-format-sft-20261115-03` | `kda-960m-lcb-format-sft` | Nov 15 | Completed (answer-only format SFT, two variants); 4.7 pass@1 on an 8-problem LiveCodeBench check |
| `kda-960m-reasoning-pilot-20261115-01` | `kda-960m-reasoning-pilot` | Nov 15 | Failed in the 32K context extension stage |
| `kda-960m-reasoning-pilot-20261115-02` | `kda-960m-reasoning-pilot` | Nov 15 | Completed: 32K context extension, then the reasoning pilot (3.8 pass@1) |
| `kda-960m-code-anneal-20261117-01` | `kda-960m-code-anneal` | Nov 17 | Failed in data preparation: one source had fewer training tokens than the recipe asked for |
| `kda-960m-code-anneal-20261117-02` | `kda-960m-code-anneal` | Nov 17–18 | Completed: 5B code tokens at 32K |
| `kda-960m-code-anneal-20b-20261117-01` | `kda-960m-code-anneal-20b` | Nov 18 | Completed: 20B code tokens at 4K |
| `kda-960m-reasoning-sft-v3-20261118-01` | `kda-960m-reasoning-sft-v3` | Nov 18 | Completed: reasoning SFT v3 (8.4 pass@1) |
| `kda-960m-merge-sft-v3-code20b-l05`, `-l10` | `modal_tools::merge` | Nov 19 | Weight merges; both broke the model |
| `kda-960m-contest-rl-20261119-01` | `kda-960m-contest-rl` | Nov 19–20 | Illegal memory access while sampling at step 64; its resumed launch hung (FlexAttention recompilation) |
| `kda-960m-contest-rl-20261120-01` | `kda-960m-contest-rl` | Nov 20 | From step 50; illegal memory access at step 62 of the run |
| `kda-960m-contest-rl-20261120-02` | `kda-960m-contest-rl` | Nov 20–21 | Completed: from step 100 to 400 (14.1 pass@1) |
| `kda-960m-contest-rl-hard-20261121-01` | `kda-960m-contest-rl-hard` | Nov 21 | Out of GPU memory at 16K tokens |
| `kda-960m-contest-rl-hard-20261121-02` | `kda-960m-contest-rl-hard` | Nov 21–22 | Stopped at step 146 of 200 (11.8 pass@1 at step 100, 12.4 at 146) |
| `hybrid-720m-loop-test-03g` | `hybrid-720m-loop-test` | Nov 21–22 | Completed: closed-gate looping conversion ([looped LM plan](LOOPED_LM.md)) |
| `kda-960m-reasoning-sft-v4-20261121-01` | `kda-960m-reasoning-sft-v4` | Nov 21–22 | Function timeout at step 900 |
| `kda-960m-reasoning-sft-v4-20261122-01` | `kda-960m-reasoning-sft-v4` | Nov 22 | Stopped after 5 minutes and relaunched with the fixed session handling |
| `kda-960m-reasoning-sft-v4-20261122-02` | `kda-960m-reasoning-sft-v4` | Nov 22 | Completed: 4,100 steps (14.0 pass@1, 26.6 pass@6) |
| `kda-960m-distill-smoke-20261122-01` | `kda-960m-distill-smoke` | Nov 22 | Completed: 3-step smoke run of on-policy distillation on 2 H100s |
| `kda-960m-distill-pilot-a/b/c-20261122-01` | `kda-960m-distill-pilot-a/b/c` | Nov 22 | First launches of the learning-rate pilots; relaunched as `-02` after scoring counted the teacher's end of turn as the student's EOS |
| `kda-960m-distill-pilot-a/b/c-20261122-02` | `kda-960m-distill-pilot-a/b/c` | Nov 22 | Completed: three learning rates. Only the lowest stayed stable (13.2 pass@1 against 14.1 for its parent) |
| `kda-960m-distill-pilot-d-20261122-01`, `-e-20261122-01` | `kda-960m-distill-pilot-d`, `-e` | Nov 22 | Completed: sharpened teacher, and the lowest rate for three passes. Looping returned as teacher KL fell |
| `kda-960m-contest-rl-v4-20261122-01` | `kda-960m-contest-rl-v4` | Nov 22 | Completed: 100 RL steps from SFT v4 (16.3 pass@1) |
| `kda-960m-contest-rl-v4-20261123-01` | `kda-960m-contest-rl-v4` | Nov 23 | Completed: from step 100 to 400 (18.1 pass@1, 25.0 pass@6) |
| Teacher comparison | `modal_livecodebench --pool train --openrouter` | Nov 23 | Five hosted teachers on 60 problems the Qwen3-30B-A3B teacher never solved: DeepSeek V4 Flash solved 28 for $0.43 at list price and was chosen (DeepSeek V4 Pro 32, DeepSeek V3.2 25, GLM-5.3 and GLM-5.3 Flash 21 each) |
| Teacher traces v2 | `modal_livecodebench --pool codecontests --openrouter` | Nov 23 | DeepSeek V4 Flash on DeepMind CodeContests problems, and the Qwen3-30B-A3B run rebuilt with up to four passing traces per problem: 31,568 traces against 2,234 in v1, in `jfrick/lcb-teacher-traces` |
| `kda-960m-reasoning-sft-v5-20261123-01` | `kda-960m-reasoning-sft-v5` | Nov 23–24 | Completed: 4,660 steps (12.8 pass@1, 23.3 pass@6) |
| `kda-960m-contest-rl-v5-20261124-01` | `kda-960m-contest-rl-v5` | Nov 24–25 | Stopped at about step 650: 15.6 pass@1 at step 100, 18.9 at 250, 17.0 at 500 |
| `kda-960m-contest-functions-rl-20261124-01` | `kda-960m-contest-functions-rl` | Nov 24–25 | Stopped at step 45: the first mixed pool gave MBPP problems without their assertions and held-out MBPP stayed at 0% |
| `kda-960m-contest-functions-rl-20261125-01` | `kda-960m-contest-functions-rl` | Nov 25 | Completed: 400 mixed RL steps from contest RL step 250 (19.1 pass@1, HumanEval 50.1, MBPP 41.6). Step 400 is Malama v0.0.1 |
| Perplexity suite | `modal_benchmarks --suite perplexity` | Nov 25 | Malama Base and eleven open base models on 1,000 arXiv abstracts of 2026-11-24 and WikiText-2: Malama Base 0.949 and 0.833 bits per byte |
| Release | | Nov 27 | Step 400 of the mixed RL run published as [`jfrick/malama-v0.0.1`](https://huggingface.co/jfrick/malama-v0.0.1); evaluation outputs, RL pools and run logs in `jfrick/lm-exploration-results` under `pr-197/` |

## Day by day

### 10 to 14 November: pretraining

- **Malama Base** (`kda-960m-foundation-20261110-01`, recipe
  `kda-foundation-100b-504010-v3`): 99,999,547,392 tokens at 4K context, 95,367 steps of
  256 sequences, on 8 H100s. 50% web text, 40% code, 10% math. Muon for matrices and
  AdamW for the rest, constant learning rate to 85B tokens, then decay over the last 15%.
- **Seven launches.** Each launch has a 24-hour limit and sessions of up to six hours, so
  the run resumed from checkpoints as attempts `a1` to `a7`. Training steps took 75.3
  hours (602 H100-hours) at a median 380.5k tokens per second.
- **Three stretches were retrained** from an earlier checkpoint: steps 9 to 8,276 of the
  first launch, after a preemption of the workflow controller opened a second attempt;
  steps 27,909 to 30,527, after a GPU preemption; and steps 85,676 to 85,972, cut off by a
  launch deadline before its final checkpoint.
- **Loss.** Training loss fell from 2.07 at the start of the decay to 1.75 at the end.
  Held-out bits per byte fell 13% to 28% per source during the decay (general web 0.823
  to 0.715, starred Python 0.564 to 0.407); between 50B and 85B tokens at constant
  learning rate they had moved by about 1% or less.
- **Published** as `jfrick/malama-base` with an export every 5B tokens.

### 14 to 15 November: fill-in-the-middle branch and the first format SFT

- **FIM mid-training** (`kda-960m-fim-midtrain-20261114-01`): continued the foundation
  run's last constant-rate checkpoint (84.8B tokens) with its optimizer, RNG and data
  cursor, rearranged half of all code documents as prefix, suffix, middle, and decayed the
  learning rate at the same token positions as the base run. About 15B tokens.
  HumanEval-Infilling: 57.1 single-line and 19.8 multi-line, against 2.9 and 0.9 for
  Malama Base, and below the dedicated code models of similar size (69 to 85 single-line).
- **LCB format SFT**, three launches. `-01` failed before training because the recipe was
  not prepared on the data volume; `-02` failed at startup because the trainer could not
  read the recipe's data files; `-03` completed both variants (answer-only contest
  programs, and the same with OpenThoughts3 answers). 4.7 pass@1 on an 8-problem
  LiveCodeBench check.
- **Reasoning pilot** (`kda-960m-reasoning-pilot`), two launches. `-01` failed in the 32K
  context extension stage. `-02` completed both stages: the context extension (2B tokens
  at 32K, half foundation replay, repository-grouped code and books) and a 537M-token
  thinking SFT on OpenCodeReasoning-2 traces. **3.8** pass@1: the first thinking model.

### 17 to 18 November: code mid-training and reasoning SFT v3

- **Code mid-training, 5B** (`kda-960m-code-anneal`). `-01` failed in data preparation:
  one source had fewer training tokens than the recipe asked for. `-02` completed 5B
  tokens at 32K from the context extension, with 45% replay. Plain completion: HumanEval
  50.0 (39.0 with docstring examples stripped), MBPP 33.2.
- **Code mid-training, 20B** (`kda-960m-code-anneal-20b-20261117-01`): 20B tokens at 4K
  from Malama Base, 35% replay, a different mixture with OpenCoder's Stack v2 files and
  code web pages. HumanEval 42.7 (31.1 stripped), MBPP 42.4. Four times the tokens raised
  MBPP by 9 points and lowered HumanEval by 7; the runs differ in start, context and
  mixture, so the token count is not isolated.
- **Decision:** reasoning SFT continues from the 5B export, which keeps the 32K context
  the thinking protocol needs.
- **Reasoning SFT v3** (`kda-960m-reasoning-sft-v3-20261118-01`): 1.42B tokens at 32K,
  one pass, 10% replay. **8.4** pass@1 (easy 32.8, medium 1.1, hard 0.1). 78.9% of
  answers ran to the token limit and 97% of those ended in a verbatim loop, starting a
  median 26% into the answer. A second SFT was not run, because more imitation does not
  target the model's own decoding falling into repetition and an outcome reward does.

### 19 November: weight merge

- **Task arithmetic** tried to add the 20B code run's change to the reasoning model:
  `v3 + λ·(code-20B − Malama Base)`, at λ = 0.5 and 1.0. Both broke the model: held-out
  loss rose from 0.95 to 3.09 on Python at λ = 0.5 and to about 13.7 on every source at
  λ = 1.0, above the 11.95 of a uniform guess. The two branches had diverged too early
  (32K context and the 5B run against the 4K base). Not evaluated further.

### 19 to 21 November: contest RL from SFT v3

- **Setup.** Dr. GRPO with Muon, 64 problems and 8 answers per step on 8 H100s, 8K-token
  answers, reward 1 or 0 from LiveCodeBench's grader in a sandbox and −0.5 for an
  unfinished answer. Pool `rl-pool-contest-02`: 1,843 training problems (1,155 medium,
  688 easy) and 56 held out.
- **Three launches for 400 steps.** `-20261119-01` stopped with a CUDA illegal memory
  access while sampling at step 64, and its resumed launch hung for two hours.
  `-20261120-01` started from step 50 and hit the same error at step 62 of its run.
  `-20261120-02` ran from step 100 to 400.
- **Two fixes.** Training rows are padded to multiples of 1,024 tokens, because padding
  each micro-batch to its longest answer made compiled attention recompile for almost
  every batch, fall back to the unfused path and run out of memory. Rollouts reuse the
  smallest cached decoder with enough slots, where they had captured a new KV cache and
  CUDA graph almost every step. The root cause of the illegal access was not proven.
- **Result:** 10.9 at step 50, 11.3 at step 100, **14.1** at step 400 (easy 46.9, medium
  6.4, hard 1.6). Truncation fell from 79% to 8% in the first 50 steps and to 0.04% by
  the end.
- **Greedy validation did not predict the benchmark.** One greedy answer per held-out
  problem looped on 30% to 80% of problems between steps 275 and 375, while sampled
  answers finished, and the final checkpoint was the best on the benchmark.

### 21 to 22 November: hard-problem RL, the loop test and SFT v4's first launches

- **Contest RL on medium and hard problems only** (`kda-960m-contest-rl-hard`, pool
  `rl-pool-contest-03`: 1,154 medium and 403 hard problems, 16K-token answers). `-01` ran
  out of GPU memory at 16K tokens; target log-probabilities are now computed in chunks of
  2,048 positions. `-02` scored 11.8 at step 100, below the 14.1 it started from, and was
  stopped at step 146 of 200 (12.4). Training reward rose from 0.36 to 0.50 while
  held-out accuracy fell from 22.2 to 18.8; easy problems, absent from the pool, lost 6 to
  11 points.
- **Closed-gate loop test** on the 720M model (`hybrid-720m-loop-test-03g`): one loop
  matched the control and two and three loops were 0.001 to 0.005 nats below it
  ([LOOPED_LM.md](LOOPED_LM.md)). Looping was not adopted for Malama.
- **Reasoning SFT v4, first launches.** `-20261121-01` hit its 5.5-hour function timeout
  at step 900: the held-out evaluation over 207 subsets took 18 minutes each time, and the
  session hung on shutdown after writing its report. The config now evaluates every 500
  steps on 4 documents per subset, a session that has written its report is ended, and a
  failed session is retried within the launch's deadline. `-20261122-01` was stopped
  after 5 minutes and relaunched with that handling as `-20261122-02`.

The days from 22 November belong to one session
([koogle/lm-exploration#197](https://github.com/koogle/lm-exploration/pull/197)). It
started from the contest RL checkpoint of SFT v3 at step 400 (14.1) with about $3,000 of
Modal credit.

### 22 November

- **Plan.** Spend most of the budget on on-policy distillation from GLM-4.7-Flash, which
  shares Malama's tokenizer; one round only, after reasoning SFT v4 finishes. SFT v4
  (`kda-960m-reasoning-sft-v4-20261122-02`, 4,100 steps) was already training.
- **Distillation pilot data.** 4,200 answers sampled from the 14.1 checkpoint (98%
  finished, about 3,000 tokens each) and scored by the teacher on 4 H100s. One scoring
  shard ran out of GPU memory on a 10K-token answer; scoring now converts teacher outputs
  in chunks of 2,048 positions. The teacher's end-of-turn token is counted as the student's
  EOS.
- **Learning-rate pilots** (`kda-960m-distill-pilot-a/b/c`, then `d` with a teacher
  sharpened to temperature 0.6 and `e` at the lowest rate for three passes), 8 H100s each,
  judged on 56 held-out contest problems:

  | Run | Teacher KL, start to end | Held-out accuracy, start to end | Truncated at end |
  | --- | --- | --- | ---: |
  | Lowest rate, 1 pass | 0.71 to 0.54 | 38.4 to 37.9 | 0.4% |
  | Lowest rate, 3 passes | 0.71 to 0.33 | 37.1 to 35.7 | 5.4% |
  | Middle rate, 1 pass | 0.66 to 0.34 | 37.5 to 32.6 | 11.6% |
  | Middle rate, sharpened teacher | 0.62 to 0.36 | 36.2 to 31.7 | 8.0% |
  | Highest rate, 1 pass | 0.57 to 0.23 | 36.6 to 0.9 | 81.7% |

  Truncated answers were verbatim loops, and looping came back as teacher KL fell,
  whatever the learning rate.
- **Lightly distilled checkpoint on the benchmark:** 13.2, against 14.1 for its parent.
  Decision: the full distillation round was not run.
- **Contest RL on medium and hard problems only** (`kda-960m-contest-rl-hard-20261121-02`,
  step 146): 12.4, below 14.1. That stage stayed stopped.
- **SFT v4 finished:** 14.0 pass@1 and 26.6 pass@6 before any RL, with 58% of answers
  running to the token limit.
- **Contest RL from v4, 100 steps** (`kda-960m-contest-rl-v4-20261122-01`): training
  truncation fell from about 45% to 3%; **16.3**.
- **Decision:** build a larger SFT v5 from more teacher traces instead of distilling, on
  the evidence that RL from v4 raised pass@1 but not pass@6 (26.6 to 26.0).

### 23 November

- **Contest RL from v4 extended** by 300 steps (`kda-960m-contest-rl-v4-20261123-01`).
  At 400 steps: **18.1** pass@1, 25.0 pass@6.
- **Teacher comparison through OpenRouter** on 60 problems the Qwen3-30B-A3B teacher had
  never solved, 4 samples each, 16K-token cap:

  | Teacher | Problems solved | Passing samples | List cost |
  | --- | ---: | ---: | ---: |
  | DeepSeek V4 Pro | 32 | 72 | $4.23 |
  | DeepSeek V4 Flash | 28 | 82 | $0.43 |
  | DeepSeek V3.2 | 25 | 53 | $1.05 |
  | GLM-5.3 | 21 | 40 | $4.52 |
  | GLM-5.3 Flash | 21 | 40 | $0.28 |

  DeepSeek V4 Flash was chosen. The account was billed about twice the list prices.
- **Teacher traces v2.** The trace job gained an OpenRouter generator and DeepMind
  CodeContests problems. DeepSeek V4 Flash solved 4,748 of the first 8,000 CodeContests
  problems (89% of easy, 74% of medium, 30% of hard), giving 16,226 passing traces; the
  existing Qwen3-30B-A3B run was rebuilt with up to four passing traces per problem
  (9,101). With the remaining CodeContests problems the set reached 31,568 traces, against
  2,234 in v1. Published to the private dataset `jfrick/lcb-teacher-traces`.
- **SFT v5 launched** (`kda-960m-reasoning-sft-v5-20261123-01`): v4's sources with the
  v2 traces at four passes, 4.89B tokens, 4,660 steps, about 12 hours on 8 H100s.

### 24 November

- **SFT v5 finished:** 12.8 pass@1 and 23.3 pass@6 before RL, below v4, with 74% of
  answers running to the token limit.
- **Contest RL from v5** (`kda-960m-contest-rl-v5-20261124-01`, checkpoint every 50
  steps): 15.6 at step 100, **18.9** pass@1 and 27.6 pass@6 at step 250, 17.0 and 24.5 at
  step 500. It peaked near step 250. The run was stopped at about step 650.
- **Function tasks in the chat protocol.** HumanEval and MBPP were scored as chat problems
  with thinking for the first time. The step-250 checkpoint scored 10.6 on HumanEval and
  2.6 on MBPP, against 56.4 and 42.6 for Qwen3-0.6B: 67% of its HumanEval answers were
  programs reading standard input where a function was asked for.
- **Plain-completion checkpoints.** HumanEval and MBPP for the code mid-training runs:
  50.0 and 33.2 after 5B tokens, 42.7 and 42.4 after 20B. No single checkpoint holds both
  highs.
- **First mixed RL pool** (`rl-pool-contest-functions-01`: contest problems, KodCode
  function problems, MBPP validation problems without their assertions), run
  `kda-960m-contest-functions-rl-20261124-01` from contest RL step 550. Held-out MBPP
  stayed at 0%; stopped after 45 steps.

### 25 November

- **Second mixed pool** (`rl-pool-contest-functions-02`): 1,843 contest problems, 879
  KodCode function problems and 374 MBPP train and validation problems stated with their
  test assertions.
- **Mixed RL** (`kda-960m-contest-functions-rl-20261125-01`, 400 steps from contest RL
  step 250, 8 H100s, about 12 hours):

  | Mixed RL step | LiveCodeBench pass@1 | pass@6 | HumanEval | MBPP |
  | --- | ---: | ---: | ---: | ---: |
  | 0 (contest RL step 250) | 18.9 | 27.6 | 10.6 | 2.6 |
  | 200 | 18.2 | 26.8 | 48.4 | 41.1 |
  | 250 | 17.5 | 25.9 | | |
  | 400 | **19.1** | 26.6 | **50.1** | **41.6** |

  Held-out validation swung widely during the run (MBPP between 3% and 42%, up to 71% of
  validation answers unfinished at step 250) while benchmark scores of the same
  checkpoints stayed stable, the pattern seen in the contest RL from v3. Step 400 became Malama v0.0.1.
- **Perplexity suite** added to the benchmark harness. Malama Base and eleven open base
  models were scored on 1,000 arXiv abstracts of 2026-11-24 and on WikiText-2: Malama Base
  0.949 and 0.833 bits per byte. Four long-context models ran out of GPU memory at eight
  documents per batch and were rerun at one.
- **Estimating a position on OpenAI's perplexity chart** from benchmark scores with
  BenchPress did not work: with only frontier anchors the fit placed weak models below
  the strongest ones.
- **OpenRouter cannot measure perplexity:** of eight models tried, none returned
  log-probabilities for supplied text.
- **Playground on Modal** (`hybrid/playground/modal_serve.py`) served the final checkpoint
  as a chat for four hours.

### 27 November

- **Release.** Weights, model card, charts and `results.json` published as
  [`jfrick/malama-v0.0.1`](https://huggingface.co/jfrick/malama-v0.0.1). Evaluation
  outputs, RL pools and run logs uploaded to `jfrick/lm-exploration-results` under
  `pr-197/`.
- **Space.** `jfrick/malama-versions` updated to open on a Chat tab with v0.0.1 as the
  default model. The Space stayed paused: Hugging Face requires pre-paid credits for its
  L4 hardware.
- **Loop test write-up.** The closed-gate looping run on the 720M model
  (`hybrid-720m-loop-test-03g`, run on 22 November by another session) was written up in
  [LOOPED_LM.md](LOOPED_LM.md) and its records uploaded. No looping run exists for the
  960M model.

- **Reasoning off.** `modal_livecodebench --protocol chat` scores checkpoints with the
  thinking protocol's samples and grading but no reasoning span. Three checkpoints of the
  v0.0.1 line, pass@1:

  | Checkpoint | HumanEval | LiveCodeBench v6 | HumanEval with reasoning | LiveCodeBench with reasoning |
  | --- | ---: | ---: | ---: | ---: |
  | SFT v5, before RL | 21.1 | 7.0 | not measured | 12.8 |
  | + 250 contest RL steps | 21.6 | 13.1 (153 of 175 problems) | 10.6 | 18.9 |
  | + 400 mixed RL steps (v0.0.1) | 27.1 | 13.1 | 50.1 | 19.1 |

  Contest RL raised LiveCodeBench without reasoning from 7.0 to 13.1 and left HumanEval
  where it was; mixed RL raised HumanEval by 5.5 points and left LiveCodeBench where it
  was. The workspace reached its Modal spend limit twice during these runs: one of eight
  LiveCodeBench shards of the contest RL checkpoint and most of MBPP did not finish (MBPP:
  7.1 for the contest RL checkpoint on 251 of 500 problems, nothing for the other two).
  The thinking-protocol HumanEval run of SFT v5 was stopped at the start in favour of
  these.

### Spend

Modal credit was about $3,000 at the start of the session and about $1,600 on 24 November
at 16:00, as reported by the account owner. The session's own estimate at the end was $350
to $400 left; the balance was not read from Modal. Eight H100s cost about $32 per hour:
SFT v5 took about 12 hours and the final mixed RL run about 12 hours. OpenRouter traces
were paid from $150 of credit added on 23 November, after about $20 for the teacher
comparison.
