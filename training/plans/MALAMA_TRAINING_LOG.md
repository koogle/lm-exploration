# Malama training log

A dated record of what was run to produce Malama v0.0.1, in the order it happened: each
run, what it measured and the decision that followed. Times are UTC. Results and
conclusions are summarized in [MALAMA_V0_0_1.md](MALAMA_V0_0_1.md) and
[CODING_REASONER_1B_FINDINGS.md](CODING_REASONER_1B_FINDINGS.md); this file keeps the
sequence, including what did not work.

Entries are grouped by day under a heading for the session that wrote them, so logs from
different sessions merge by placing their day sections in date order.

## 22 to 27 November 2026: distillation pilots, SFT v4 and v5, contest and mixed RL, release

Session behind [koogle/lm-exploration#197](https://github.com/koogle/lm-exploration/pull/197).
It started from the contest RL checkpoint of reasoning SFT v3 at step 400 (14.1 pass@1 on
LiveCodeBench v6 with thinking), with about $3,000 of Modal credit. Scores below are
LiveCodeBench v6 pass@1 in the thinking protocol unless stated.

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
  checkpoints stayed stable. Step 400 became Malama v0.0.1.
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

### Spend

Modal credit was about $3,000 at the start of the session and about $1,600 on 24 November
at 16:00, as reported by the account owner. The session's own estimate at the end was $350
to $400 left; the balance was not read from Modal. Eight H100s cost about $32 per hour:
SFT v5 took about 12 hours and the final mixed RL run about 12 hours. OpenRouter traces
were paid from $150 of credit added on 23 November, after about $20 for the teacher
comparison.
