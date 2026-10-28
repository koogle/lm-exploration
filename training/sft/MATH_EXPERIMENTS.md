# Math SFT experiments (October 25, 2026)

Answer-only SFT on GSM8K + MetaMathQA word problems ([README](README.md)) from
two end points of the 20B WSD pretraining run
([results](../plans/SIMPLE_LLM_PRETRAINING.md#results-2026-10-25)). Single seed, 124M.

## Runs

Data `math-gsm8k-metamath-02` (dataset SHA256 `f757e05f…`): 224,035 training rows
(7,469 GSM8K, 217,064 MetaMathQA GSM rows) and 498 GSM8K validation problems; 17,015
rewrites of validation problems and 97 test-overlapping problems were dropped. Both runs
used [`recipes/`](recipes/): 7,500 steps of 32 examples (about one epoch), context 2,048,
constant LR 5e-5 with AdamW, one H100.

| Run ID | Parent | Validation loss, step 500 → 7,500 |
| --- | --- | --- |
| `math-sft-metamath-20b-r01` | 20B end point, `climbmix-124m-wsd20b-02-s02/step-000203724.pt` | 1.147 → 0.948 |
| `math-sft-metamath-5b-r01` | 5.45B end point, `climbmix-124m-wsd20b-02-decay5b-s01/step-000055555.pt` | 1.215 → 1.008 |

Validation loss is answer-token cross-entropy and was still falling at the end.

## Results

[`analysis/evaluation`](../analysis/evaluation/README.md), final checkpoints
(`step-000007500.pt`). Math is greedy; base models use 5-shot exemplars, SFT models the
zero-shot training format. GSM8K is the first 500 test problems; SVAMP all 300.

| Model | GSM8K | SVAMP | ClimbMix bpb | Code bpb | HellaSwag | ARC-E | ARC-C | OBQA | LAMBADA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.45B base | 1.8 | 1.0 | 0.913 | 0.536 | 34.3 | 55.3 | 27.0 | 31.2 | 25.5 |
| 5.45B + math SFT | 15.8 | 14.3 | 1.201 | 1.341 | 31.6 | 43.8 | 23.9 | 26.6 | 19.8 |
| 20B base | 1.8 | 1.0 | 0.876 | 0.501 | 38.0 | 57.3 | 28.3 | 32.6 | 28.6 |
| 20B + math SFT | 20.4 | 20.3 | 1.178 | 1.182 | 33.6 | 46.2 | 24.9 | 29.0 | 24.2 |

- SFT turns near-zero math into 15–20% on GSM8K and SVAMP. The SVAMP gain is on a test
  set none of the training data came from.
- More pretraining transfers: the 20B parent ends 4.6 points higher on GSM8K and 6 on
  SVAMP, with lower validation loss throughout.
- The cost is forgetting. Held-out ClimbMix bits per byte rose about 35%, code bits per
  byte more than doubled, and ARC-Easy fell 11 points. Every update saw only math.
- Answers follow the trained format (steps, then `#### N`). Errors are mostly wrong setups
  or arithmetic, for example `15 - 19 = 2` for fish that disappeared.

Result JSONs (including every generated answer) are in `jfrick/lm-exploration-results`
under `pr-76/evaluation/`. A first SVAMP scoring took the first number of at most 32
tokens and scored the SFT models near 1%; the scores above take the `####` answer, else
the last number (commit `e95ce4d`). The evaluation now checks answers with Math-Verify;
these scores have not been recomputed with it.

## Replay runs (r02)

[`recipes/math-metamath-replay-{20b,5b}.json`](recipes/) train the same data from the
same parents for 15,000 steps (about two epochs) with a cosine schedule (2e-5 peak, 300
warmup steps, decaying to 2e-6), and replay 8 windows of 1,024 pretraining tokens per
step from the 20B run's snapshot: 60% ClimbMix, 40% code (35% curated, 5% MIT), with
replay weight 0.25. Each run took about 1h54 on one H100. r02 ran with replay built into
`trainer/sft.py` (commit `99b1ed5`), which started each window at a random stored
document of its source. `trainer/replay_hook.py` gives the same weighting, loss and
validation, and reads each source in order through the pretraining loader.

Final checkpoints (`step-000015000.pt`) with r01 rescored in the same call
(`math-sft-r01-r02-01`), math checked with Math-Verify, zero-shot:

| Model | GSM8K | SVAMP | ClimbMix bpb | Code bpb | HellaSwag | ARC-E | ARC-C | OBQA | LAMBADA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.45B + math SFT (r01) | 15.8 | 14.3 | 1.201 | 1.341 | 31.6 | 43.8 | 23.9 | 26.6 | 19.8 |
| 5.45B + math SFT + replay (r02) | 14.6 | 9.7 | 0.926 | 0.634 | 34.4 | 49.3 | 25.9 | 29.8 | 25.1 |
| 20B + math SFT (r01) | 20.4 | 20.3 | 1.178 | 1.182 | 33.6 | 46.2 | 24.9 | 29.0 | 24.2 |
| 20B + math SFT + replay (r02) | 17.4 | 16.7 | 0.890 | 0.618 | 38.2 | 53.1 | 26.3 | 32.8 | 29.1 |

- Replay removes most of the forgetting. From the 20B parent, ClimbMix bits per byte end
  1.6% above the base model (r01: 35%); HellaSwag, OpenBookQA and LAMBADA match or exceed
  the base model, and ARC-E is 4 points below it (r01: 11).
- Code bits per byte still end 23% above the base model (r01: 136%), although code is
  40% of the replay; replay validation loss on curated code rose steadily through training.
- Math is 3 points lower on GSM8K and 4 on SVAMP than r01 for the 20B parent; the 5.45B
  parent loses 5 points on SVAMP. The r02 answer loss ends slightly higher (0.972 vs 0.948).
- Math-Verify rescoring leaves r01 unchanged (20.4 / 20.3 and 15.8 / 14.3).

The evaluation result (with every generated answer) and the r02 training metrics and
manifests are in `jfrick/lm-exploration-results` under `pr-87/evaluation/` and
`pr-87/training/`.

Three-step resume qualifications on H100 (`qualify --steps 3`) did not match bit for bit,
with replay and with the r01 recipe without it: step, sampling and RNG state matched,
model and optimizer weights did not. Math runs take at most about two hours, so an
interrupted run is rerun from scratch rather than resumed.

## Next

Reinforcement learning on math and code with verifiable rewards, after a joint math and
code SFT stage: see [RL](../rl/README.md).
