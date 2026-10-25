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

## Replay runs (r02, in progress)

[`recipes/math-metamath-replay-{20b,5b}.json`](recipes/) train the same data from the
same parents for 15,000 steps (about two epochs) with a cosine schedule (2e-5 peak, 300
warmup steps, decaying to 2e-6), and replay 8 windows of 1,024 pretraining tokens per
step from the 20B run's snapshot: 60% ClimbMix, 40% code (35% curated, 5% MIT), with
replay weight 0.25. Three-step resume qualifications on H100 (`qualify --steps 3`) did
not match bit for bit, with replay and with the r01 recipe without it: step, sampling and
RNG state matched, model and optimizer weights did not. Math runs take under an hour, so
an interrupted run is rerun from scratch rather than resumed. Results will be compared with r01 on math and general benchmarks.

## Next

Reinforcement learning on GSM8K-style problems with a verifiable answer reward, starting
from the better replay checkpoint.
