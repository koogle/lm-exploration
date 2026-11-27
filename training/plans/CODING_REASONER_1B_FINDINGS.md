# Sub-1B coding reasoner: findings, November 2026

Measured results of the stages after foundation pretraining in the
[training plan](CODING_REASONER_1B.md): code mid-training, thinking-mode reasoning SFT
(v3 and v4), a weight-merge attempt and contest RL. The target is LiveCodeBench v6 in thinking mode,
where MobileLLM-R1.5-950M reports 29.0 pass@1 (a reference score only; none of its
outputs are used). Result files are in the HF dataset
[`jfrick/lm-exploration-results`](https://huggingface.co/datasets/jfrick/lm-exploration-results),
folder `training/results/coding-reasoner-1b/`: `runs/` (each launch's workflow record,
stage records, run configs, metrics, held-out losses, reports and logs), `lcb-thinking/`
(per-problem LiveCodeBench scores and our models' completions), `code/`, `merge/` and
`pools/`. The [run log](#run-log) lists every launch.

## Code mid-training: 5B and 20B tokens

Two code-heavy continuations of the 960M KDA hybrid
([`kda-960m-code-anneal.json`](../workflows/kda-960m-code-anneal.json),
[`kda-960m-code-anneal-20b.json`](../workflows/kda-960m-code-anneal-20b.json)), scored
zero-shot on HumanEval (with and without the docstring examples, "stripped") and 3-shot
on MBPP, pass@1:

| Run | Start | Tokens | Replay | HumanEval | HumanEval stripped | MBPP |
|---|---|---|---|---|---|---|
| `kda-code-anneal-v1` | 32K context extension | 5B at 32K | 45% | **50.0** | **39.0** | 33.2 |
| `kda-code-anneal-v2` | Malama Base (4K) | 20B at 4K | 35% | 42.7 | 31.1 | **42.4** |

Four times the code tokens did not make a better model on both tests: HumanEval fell 7
to 8 points and MBPP rose 9. The two runs differ in more than length (starting
checkpoint and context, mixture version with OpenCoder's Stack v2 files and code web
pages in v2, replay share), so this does not isolate the token count. The reasoning SFT
continued from the 5B export, which keeps the 32K context the thinking protocol needs.

## Reasoning SFT v3

[`kda-960m-reasoning-sft-v3.json`](../workflows/kda-960m-reasoning-sft-v3.json):
1.42B tokens of `chat-sft-reasoning-v3` at 32K (one pass, 10% replay), from the 5B code
export; next-token loss on assistant tokens. LiveCodeBench v6, thinking protocol (16
samples per problem at temperature 0.6, up to 32,768 new tokens):

| | pass@1 | Easy | Medium | Hard | Truncated | Mean tokens |
|---|---|---|---|---|---|---|
| Reasoning pilot | 3.8 | | | | | |
| Reasoning SFT v3 | 8.4 | 32.8 | 1.1 | 0.1 | 78.9% | 26.2K |

Most answers never finish, and nearly all of those are loops: of the 2,208 truncated
samples (of 2,800), 97% end in a block of text repeated at least three times verbatim
(`livecodebench.verbatim_loop`, reported by the merge since this run), with the loop
starting at a median 26% into the answer and a median period of 424 characters. A
second, short SFT on the same next-token objective was not run: the failure is the
model's own decoding falling into repetition, which more imitation does not target and
an outcome reward does.

## Weight merge: negative result

Task arithmetic tried to add the 20B code run's change to the reasoning SFT:
`v3 + λ·(code-20B − Malama Base)` (`model_hybrid.modal_tools::merge`, branch
`jf-claude/weight-merge`). Held-out loss at 32K (100 documents per source, the
foundation recipe's held-out sets):

| Source | v3 | λ = 0.5 | λ = 1.0 |
|---|---|---|---|
| Python | 0.95 | 3.09 | 13.71 |
| Code | 1.01 | 3.27 | 13.68 |
| ClimbMix | 2.52 | 4.24 | 13.64 |
| Math | 1.76 | 3.54 | 13.56 |

Both merges break the model; at λ = 1.0 the loss is near uniform over the 154,880-token
vocabulary (ln 154,880 = 11.95 is uniform; above it the model is confidently wrong). The
two lines diverged early: v3 descends from the 32K context extension and the 5B run, the
20B run from the 4K base, so the 20B delta is not a direction in v3's neighborhood. The
merges were not evaluated further.

## Contest RL

[`kda-960m-contest-rl.json`](../workflows/kda-960m-contest-rl.json) (described in the
[RL guide](../rl/README.md#contest-problems-in-thinking-mode)): Dr. GRPO with Muon from
reasoning SFT v3, 64 problems × 8 answers per step on 8 H100s, 8K-token answers,
reward 1 or 0 from LiveCodeBench's grader in the sandbox and −0.5 for an unfinished
answer. Pool `rl-pool-contest-02`: 1,843 training problems (1,155 medium, 688 easy;
LeetCode-heavy, 10% AtCoder) and 56 held out, screened against LiveCodeBench v6.

The 400 steps ran as three launches. Two stopped with a CUDA illegal memory access while
sampling, at steps 64 and 62 of their runs; each later launch started a new run from the
previous run's last checkpoint (steps 50 and 100), with a new optimizer state:

| Run | Cumulative steps | End |
|---|---|---|
| `kda-960m-contest-rl-20261119-01` | 0–64 | crash; resumed run hung (below) |
| `kda-960m-contest-rl-20261120-01` | 50–112 | crash |
| `kda-960m-contest-rl-20261120-02` | 100–400 | completed, exported to `jfrick/malama-contest-rl` |

Two fixes made the last run complete:

- **Compiled attention widths** (commit `41e0477`): the trainer padded each micro-batch
  to its longest answer, so compiled FlexAttention recompiled for almost every batch and,
  after eight shapes, fell back to the unfused path that materializes the full score
  matrix. GPUs ran out of memory and the resumed first run hung for two hours. Training
  rows are now padded to multiples of 1,024 tokens, and the hybrid family raises the
  recompile limit to 64.
- **Decoder reuse** (commit `9949f11`): rollouts keep 64 rows, but the decoder's key
  included the cache capacity (prompt width + 8,192), so with two cached decoders a new KV
  cache and CUDA graph were captured and the old ones freed almost every step, under
  `expandable_segments`. A call now reuses the smallest cached decoder with enough slots.
  The output is unchanged (H100 check in `tests/check_hybrid_cuda.py`). The root cause of
  the illegal access is not proven; the run with this fix passed step 64 and completed
  300 steps.

Training, means over each span (cumulative steps; KL is measured from each run's start):

| Steps | Reward | Truncated | Answer tokens | Sampled NLL | KL |
|---|---|---|---|---|---|
| 1–10 | −0.08 | 57% | 6,220 | 0.68 | 0.00 |
| 41–50 | 0.30 | 0.4% | 2,110 | 0.36 | 0.04 |
| 101–110 | 0.39 | 0.2% | 2,570 | 0.28 | 0.00 |
| 171–200 | 0.47 | 0.2% | 1,980 | 0.28 | 0.08 |
| 261–290 | 0.53 | 1.5% | 3,330 | 0.28 | 0.19 |
| 381–400 | 0.58 | 0.1% | 2,470 | 0.29 | 0.31 |

Validation (one greedy answer on each of the 56 held-out problems):

| Step | 0 | 25 | 50 | 75 | 100 | 150 | 200 | 250 | 275 | 300 | 350 | 375 | 400 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 21.4 | 37.5 | 30.4 | 39.3 | 50.0 | 46.4 | 37.5 | 42.9 | 23.2 | 21.4 | 8.9 | 17.9 | 35.7 |
| Truncated | 64% | 23% | 5% | 7% | 11% | 5% | 7% | 11% | 63% | 52% | 80% | 30% | 4% |

LiveCodeBench v6, thinking protocol:

| | pass@1 | Easy | Medium | Hard | LeetCode | AtCoder | Truncated | Looping | Mean tokens |
|---|---|---|---|---|---|---|---|---|---|
| Reasoning SFT v3 | 8.4 | 32.8 | 1.1 | 0.1 | 6.9 | 9.3 | 78.9% | 76.7% | 26.2K |
| RL step 50 | 10.9 | 41.4 | 2.2 | 0.1 | 9.1 | 11.8 | 8.3% | 8.2% | 5.5K |
| RL step 100 | 11.3 | 41.3 | 3.4 | 0.2 | 8.4 | 12.8 | 7.5% | 7.3% | 5.5K |
| RL step 400 | **14.1** | 46.9 | 6.4 | 1.6 | 10.1 | 16.4 | 0.04% | 0.04% | 3.4K |
| MobileLLM-R1.5-950M (reported) | 29.0 | | | | | | | | |

What this shows:

- RL removed the looping first: by step 50, truncation fell from 79% to 8% and pass@1
  rose 2.5 points, mostly on easy problems. The later 300 steps added 2.8 points, with
  medium (3.4 → 6.4) and hard (0.2 → 1.6) improving on both platforms.
- Greedy validation did not predict the benchmark. From step 275 to 375 greedy answers
  looped on 30–80% of the held-out problems while sampled training answers at temperature
  1.0 finished (truncation ≤ 1.5%), and the final checkpoint, below steps 100–250 on
  greedy validation, is the best on the benchmark at temperature 0.6. With 56 problems, one
  answer each, a problem is 1.8 points and the standard error is about 6 points.
- Training reward kept rising (0.39 → 0.58 over the last run) while the KL from that
  run's start reached 0.3; at the end, 633 of the 1,843 problems were resting (their 8
  answers had the same reward twice in a row), most of them presumably the solved easy
  ones.
- Answers got shorter than the benchmark allows: 3.4K tokens on average with 32K
  available. Hard problems may need longer reasoning than the 8K training limit rewards.

## Contest RL on medium and hard problems: negative result

[`kda-960m-contest-rl-hard.json`](../workflows/kda-960m-contest-rl-hard.json) continued
from step 400 on medium and hard problems only (`rl-pool-contest-03`: 1,154 medium and 403
hard training problems, 44 held out; hard problems limited to those the teacher solved),
with 16K-token answers and validation on 4 sampled answers per problem at temperature 0.6.

The first launch ran out of GPU memory at 16K tokens (rank 7 aborted). Computing the
target log-probabilities in chunks of 2,048 positions instead of from the full logits
(`HybridLM.target_logprobs`, commit `41efe89`) fixed it. The second launch was stopped at
step 146 of 200, after its step-100 checkpoint scored below the starting model; it saved a
checkpoint and paused, and its deadline supervisor later closed the launch.

| Steps | Reward | Truncated | Answer tokens | KL |
|---|---|---|---|---|
| 1–10 | 0.36 | 0% | 3,290 | 0.00 |
| 41–50 | 0.45 | 0% | 2,870 | 0.04 |
| 91–100 | 0.45 | 0% | 3,240 | 0.11 |
| 137–146 | 0.50 | 0% | 3,740 | 0.17 |

| Step | 0 | 25 | 50 | 75 | 100 | 125 |
|---|---|---|---|---|---|---|
| Validation accuracy (4 samples) | 22.2 | 26.7 | 22.7 | 22.2 | 19.3 | 18.8 |
| Solved by any of the 4 | 38.6 | 38.6 | 34.1 | 34.1 | 29.5 | 31.8 |

| LiveCodeBench v6 | pass@1 | Easy | Medium | Hard | Truncated |
|---|---|---|---|---|---|
| RL step 400 (start) | 14.1 | 46.9 | 6.4 | 1.6 | 0.04% |
| + 100 steps | 11.8 | 36.2 | 7.1 | 1.6 | 3.6% |
| + 146 steps | 12.4 | 41.0 | 5.6 | 1.4 | 1.2% |

Training reward rose while held-out accuracy and the benchmark fell. Medium and hard did
not improve, and easy problems, absent from the pool, lost 6 to 11 points. The model was
short of capability on these problems more than of RL signal, which led to the larger
distillation below.

## Reasoning SFT v4

[`kda-960m-reasoning-sft-v4.json`](../workflows/kda-960m-reasoning-sft-v4.json): 4.30B
tokens of `chat-sft-reasoning-v4` at 32K (about three times v3's unique traces, two
passes, 10% replay) from the same 5B code export as v3, 4,100 steps in one 10-hour
session on 8 H100s. Held-out loss (mean over the recipe's sources): 1.401 at step 0, 0.774
at 500, 0.706 at 1,500, 0.674 at 2,500, 0.656 at 3,500.

LiveCodeBench v6, thinking protocol; pass@6 is unbiased from the 16 samples:

| | pass@1 | Easy | Medium | Hard | pass@6 | Medium @6 | Truncated | Looping |
|---|---|---|---|---|---|---|---|---|
| Reasoning SFT v3 | 8.4 | 32.8 | 1.1 | 0.1 | 14.8 | 5.4 | 78.9% | 76.7% |
| RL step 400 (from v3) | 14.1 | 46.9 | 6.4 | 1.6 | 21.3 | 13.1 | 0.04% | 0.04% |
| Reasoning SFT v4 | 14.0 | 46.7 | 7.1 | 0.9 | **26.6** | **21.3** | 57.9% | 50.0% |

- Three times the distillation data, without RL, matches 400 RL steps on pass@1 and beats
  them on pass@6. It solves 57 of the 175 problems at least once (14 medium, 6 hard).
- Looping remains: 58% of answers reach the 32K limit, 86% of those in a verbatim loop
  (median onset 57% into the answer, median period 365 characters). RL removed this
  failure from v3 within 50 steps, so RL from v4 is the next step.
- The benchmark took 82 minutes per shard (mean answer 23.7K tokens).

The first launch failed on its 5.5-hour function timeout at step 900: rank 0's held-out
evaluation over 207 subsets took 18 minutes at every evaluation, the final evaluation
repeated the last one, and the session hung on shutdown after writing its report. The
config now evaluates every 500 steps on 4 documents per chat subset, the trainer reuses
an evaluation already made at the final step, the workflow ends a session that has
written its report and does not exit (`workflows.sessions.run_session`), and the run is
one 14-hour session. A failed session is now retried within the launch's deadline when
it saved a new checkpoint (`session_retries`, [workflows README](../workflows/README.md)).

## Run log

Every launch of this study, with the later launches behind the first release, is in the
[Malama training log](MALAMA_TRAINING_LOG.md#run-log), by workflow run ID. Records of
this study's launches are under `runs/<run ID>/` in the archive.

Later stages of this line (on-policy distillation pilots, RL from SFT v4, SFT v5 and the
mixed RL behind the first release) are in [MALAMA_V0_0_1.md](MALAMA_V0_0_1.md).
