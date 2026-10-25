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
   sampler–learner KL term. Pretraining replay (`replay_*`, as in SFT) adds a
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

## Run

From `training/`:

```sh
uv run --all-extras modal run -m modal_scripts.modal_rl_train --recipe RECIPE.json
```

A recipe names `run_id`, `parent` (an SFT checkpoint, `run/step-N.pt` on the run
volume), `pool` (an `rl_pool` data ID) and `settings` (`trainer.rl.Settings`). RL
checkpoints (format `rl-v1`) keep the model code and tokenizer beside them, so
`analysis/evaluation` loads them like SFT checkpoints.
