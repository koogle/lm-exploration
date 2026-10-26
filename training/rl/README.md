# RL for math and code

Reinforcement learning for our 124M models on problems whose answers can be checked
automatically: math word problems, checked against the reference answer, and Python
functions, checked by running unit tests. The model writes complete answers and is
trained on its own samples, scored by those checkers. In SFT
([`../sft/README.md`](../sft/README.md)) it only imitates answers someone else wrote.
The problems, their checkers and the code sandbox are described in
[`../sft/CODE_DATA.md`](../sft/CODE_DATA.md).

RL can only reinforce answers the model already produces sometimes: a prompt the model
never solves gives no learning signal. So RL starts from an SFT checkpoint that has
learned the answer formats (math and code, `trainer/sft.py` with `"task": "mixture"`).

**Status:** stage 0 SFT and three RL runs are done ([results](#results-2026-10-26)): RL
raised math slightly and did not measurably change code, with no forgetting.

## Starting point and targets

Measured so far ([`../sft/MATH_EXPERIMENTS.md`](../sft/MATH_EXPERIMENTS.md)): math SFT
with pretraining replay (r02, 20B parent) scores 17.4 on GSM8K and 16.7 on SVAMP and
keeps most general ability; no model had been trained to write functions.

| Measure | Now (best SFT) | Target after RL |
| --- | --- | --- |
| GSM8K (500 test, greedy, Math-Verify) | 17.4 | ≥ 30 |
| SVAMP (300, greedy) | 16.7 | ≥ 25 |
| MBPP+ / HumanEval+ greedy pass@1 | stage 0 pending | clearly above stage 0 |
| ClimbMix bits per byte | 0.890 (base 0.876) | within 3% of base |
| HellaSwag, ARC-E | 38.2, 53.1 | within 2 points of the starting checkpoint |

These are aspirations for choosing settings, not predictions. For scale: GPT-3 175B
fine-tuned on GSM8K reached about 33%.

## Stage 0: joint SFT

[`../sft/recipes/math-code-replay-20b.json`](../sft/recipes/math-code-replay-20b.json):
the 20B end point, 20,000 steps of 32 on `mixture-math-code-02` (the 224,533 math rows
and the Qwen3.5-solved code rows, [`../sft/CODE_DATA.md`](../sft/CODE_DATA.md)), r02's
cosine schedule and pretraining replay.

## The loop

[`../trainer/rl.py`](../trainer/rl.py) on one H100. Each step samples with the current
weights and takes one optimizer step, so the loop is on-policy:

1. **Prompts.** Draw `prompts_per_step` (64) training problems of the pool,
   `math_share` (half) math. A prompt solved by none or all of its group on two visits in
   a row rests for `rest_steps` (50) steps.
2. **Rollouts.** `samples_per_prompt` (8) answers per prompt at temperature 1, without
   top-k or top-p, so samples follow the policy's probabilities: at most 256 new tokens
   for math and 384 for code. [`../trainer/rollout.py`](../trainer/rollout.py) samples a
   left-padded batch with a key/value cache. The default `graph` engine replays one
   CUDA graph per token built from custom decode kernels
   ([`../kernels/README.md`](../kernels/README.md#rollout-decoding)), 3.2 times faster
   than eager decoding on an H100 with the same error against the learner; the `eager`
   engine's greedy output matches unbatched, uncached decoding exactly
   (`tests/test_rollout.py`). The sampler also records each token's log-probability and
   its top-32 tokens.
3. **Rewards** ([`../data/rl_tasks.py`](../data/rl_tasks.py)). 1 or 0; an answer
   without EOS within the limit scores 0. Code answers are checked in the sandbox in a
   background thread while math is sampled.
4. **Advantages.** Reward minus the mean reward of the prompt's group, with no division
   by the group's standard deviation (Dr. GRPO, arXiv:2503.20783). Groups whose answers
   all score the same have zero advantage and are skipped; their share is logged.
5. **Loss.** −advantage × score summed over answer tokens, divided by the number of
   answers times the task's token limit (no per-answer length normalization). The
   samples come from the weights being trained, so no importance ratio or clipping is
   needed. The score is Score Centering's (`score_centering_top_k`, default 32;
   [`../trainer/policy_loss.py`](../trainer/policy_loss.py), the method behind Tinker's
   StableReinforce, [reference](../references/stable-reinforce-2026-10-25.md)):
   ∇log p(y) − Σ_v q_v ∇log p_v, where q is the sampler's distribution from its top-32
   with the rest of the mass spread like the learner's. On the logits its gradient is
   onehot(y) − q instead of REINFORCE's onehot(y) − p (`tests/test_policy_loss.py`).
   The sampler (cached one-token decoding) and the learner (a full forward pass) compute
   probabilities with different kernels; centering cancels any bias between them. That
   difference is logged every step as `sampler_mismatch`: in smoke runs from the r02
   math checkpoint it averaged about 1e-4 nats per token, with a mean absolute
   difference of about 0.015 per token. `score_centering_top_k: 0` gives plain
   REINFORCE.
6. **Anchors.** KL(policy ‖ parent) is estimated (k3) on the answer tokens every step and
   added with `kl_coef` (default 0); `sampler_kl_coef` (default 0) adds Tinker's
   sampler–learner KL term. Pretraining replay (`replay_*`, the hook SFT also uses,
   [`../trainer/replay_hook.py`](../trainer/replay_hook.py)) adds a
   next-token loss on pretraining windows at `replay_weight`.
7. **Optimizer.** AdamW without weight decay, LR 1e-6 after 20 warmup steps, gradient
   clipping at 1.0.

Validation is greedy accuracy on the pool's validation problems plus replay validation
loss, at the start, every `evaluate_every` steps and at the end.

### 24-hour limit

`modal_scripts/modal_rl_train.py` records a deadline 24 hours after launch and spawns the
training call (21-hour timeout, no retries; `max_seconds` at most 20 hours) and a
supervisor call. Training refuses work after the deadline and stops taking updates 30
minutes before it. The sandbox ID is written to `/runs/rl-launches/<run-id>.json`; the
supervisor cancels the training call at the deadline and terminates the sandbox when the
call ends. A run does not resume; an interrupted run is rerun from scratch.

### Logged

Every step: reward, share of uniform groups, answer length and truncation rate per task;
answers trained on; mean NLL of the sampled tokens; KL to the parent; sampler–learner
mismatch; replay loss; gradient norm; resting prompts; generation and training seconds.
With the graph engine at 64 prompts × 8 answers, a smoke run took 3.8 s to generate and
0.9 s to train per step. Every evaluation: validation accuracy and truncation per task,
and replay validation loss per source.

Signs to watch: rewards rising while EvalPlus pass@1 does not (reward hacking), sampled
NLL collapsing toward zero while reward plateaus (lost exploration), truncation rising.

## Evaluation

[`analysis/evaluation`](../analysis/evaluation/README.md) adds MBPP+ (378) and
HumanEval+ (164) greedy pass@1 in the code sandbox, next to GSM8K, SVAMP, the general
benchmarks and held-out bits per byte. Reference solutions of a 40-problem sample of
each pass all their tests through this harness.

## Runs

[`recipes/`](recipes/), all from the stage 0 checkpoint on `rl-pool-math-code-03`, with
replay at weight 0.1:

| Run | Purpose | Success looks like |
| --- | --- | --- |
| S0 | Joint math + code SFT with replay | Code pass@1 above 0, math and general scores at r02 level |
| R1 (`r1-math`) | Math-only RL, 300 steps | GSM8K rises; validates the loop cheaply |
| R2 (`r2-code`) | Code-only RL, 300 steps | MBPP+ pass@1 rises; validates the sandbox reward |
| R3 (`r3-joint`) | Joint RL, 1,000 steps | Both rise; general scores within the targets |

## Results (2026-10-26)

Single seed. Benchmarks from [`analysis/evaluation`](../analysis/evaluation/README.md),
zero-shot greedy: GSM8K first 500 test problems, SVAMP 300, MBPP+ 378 and HumanEval+ 164
pass@1; bits per byte on the snapshot's held-out windows.

| Model | GSM8K | SVAMP | MBPP+ | HumanEval+ | ClimbMix bpb | Code bpb | HellaSwag | ARC-E | LAMBADA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20B base | 1.8 | 1.0 | — | — | 0.876 | 0.501 | 38.0 | 57.3 | 28.6 |
| Math SFT + replay (r02) | 17.4 | 16.7 | — | — | 0.890 | 0.618 | 38.2 | 53.1 | 29.1 |
| S0: math + code SFT + replay | 16.0 | 13.3 | 11.6 | 9.1 | 0.891 | 0.626 | 38.0 | 54.2 | 29.4 |
| R1: S0 + 300 math RL steps | 17.0 | 13.7 | 11.1 | 11.0 | 0.891 | 0.627 | 38.0 | 54.0 | 29.9 |
| R2: S0 + 300 code RL steps | 16.2 | 13.7 | 11.6 | 9.8 | 0.891 | 0.628 | 38.0 | 54.0 | 29.7 |
| R3: S0 + 1,000 joint RL steps | 17.6 | 15.7 | 11.9 | 10.4 | 0.891 | 0.628 | 38.1 | 53.6 | 29.9 |

- Stage 0 writes working functions where no earlier model could (11.6% MBPP+, 9.1%
  HumanEval+ against the EvalPlus tests) and keeps r02's general scores; half its rows
  being code costs 1.4 GSM8K and 3.4 SVAMP points against r02. It trained for 20,000
  steps (about 2.5 hours on one H100).
- RL raised training rewards steadily (math 22% → 30%, code 27% → 31% over R1 and R2)
  but moved the benchmarks little: R3 gains 1.6 GSM8K and 2.4 SVAMP points, about one
  standard error each, and code within one or two problems. General scores and bits per
  byte did not move; KL to stage 0 stayed near 0.01 and the sampler–learner mismatch
  near 1e-4 nats per token.
- About 65% of code prompt groups (38% for math) were all right or all wrong, so they
  gave no learning signal, and up to about 900 code problems rested at a time.
- A math step took about 2.8 s and a code step about 11 s, mostly waiting on the
  sandbox's tests.

Result files (evaluations with every generated answer, training metrics and manifests,
data reports) are in `jfrick/lm-exploration-results` under `pr-87/`. The checkpoints can
be compared in the private playground Space (`jfrick/climbmix-124m-playground`).

## Next

1. Code problems matched to stage 0: keep those it solves sometimes but not always with 8
   samples, so most groups carry signal.
2. Larger learning rates (3e-6, 1e-5): KL stayed small at 1e-6.
3. Check code answers on more sandbox CPUs in parallel.
4. Repeat the best setting with a second seed before claiming a gain.

## Run

From `training/`, after the data of [`../sft/CODE_DATA.md`](../sft/CODE_DATA.md):

```sh
uv run --all-extras modal run -m modal_scripts.modal_sft_train::main --recipe sft/recipes/math-code-replay-20b.json
uv run --all-extras modal run -m modal_scripts.modal_rl_train --recipe rl/recipes/r1-math.json
```

To play with the stage 0 and RL checkpoints next to the base and math SFT models, export
them to the private comparison Space ([`../analysis/playground`](../analysis/playground/README.md)):

```sh
uv run --all-extras modal run -m playground.modal_export \
  --config analysis/playground/configs/post-training.json --name EXPORT-NAME
```

A recipe names `run_id`, `parent` (an SFT checkpoint, `run/step-N.pt` on the run
volume), `pool` (an `rl_pool` data ID) and `settings` (`trainer.rl.Settings`). RL
checkpoints (format `rl-v1`) keep the model code and tokenizer beside them, so
`analysis/evaluation` loads them like SFT checkpoints.
