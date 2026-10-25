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
   left-padded batch with a key/value cache; its greedy output matches unbatched,
   uncached decoding exactly (`tests/test_rollout.py`).
3. **Rewards** ([`../data/rl_tasks.py`](../data/rl_tasks.py)). 1 or 0; an answer
   without EOS within the limit scores 0. Code answers are checked in the sandbox in a
   background thread while math is sampled.
4. **Advantages.** Reward minus the mean reward of the prompt's group, with no division
   by the group's standard deviation (Dr. GRPO, arXiv:2503.20783). Groups whose answers
   all score the same have zero advantage and are skipped; their share is logged.
5. **Loss.** −advantage × log p(token) summed over answer tokens, divided by the number
   of answers times the task's token limit (no per-answer length normalization). The
   samples come from the weights being trained, so no importance ratio or clipping is
   needed.
6. **Anchors.** KL(policy ‖ parent) is estimated (k3) on the answer tokens every step and
   added with `kl_coef` (default 0). Pretraining replay (`replay_*`, the hook SFT also uses, [`../trainer/replay_hook.py`](../trainer/replay_hook.py)) adds a
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
answers trained on; mean NLL of the sampled tokens; KL to the parent; replay loss; gradient
norm; resting prompts; generation and training seconds. Every evaluation: validation
accuracy and truncation per task, and replay validation loss per source.

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
