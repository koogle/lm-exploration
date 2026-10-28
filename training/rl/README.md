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

## Code layout

[`../trainer/rl/`](../trainer/rl/__init__.py) keeps the parts of RL apart, so an
algorithm or a sampling regime can change without touching the rest:

| File | Owns |
| --- | --- |
| `settings.py` | What a recipe sets, in sections: `rollouts`, `algorithm`, `optimizer`, plus the shared `replay_*` fields |
| `rollouts.py` | Which prompts are drawn, how answers are sampled and rewarded |
| `algorithms.py` | How rewards become advantages and a loss; `ALGORITHMS` by name; on- or off-policy |
| `loop.py` | The training loop, checkpoints and provenance |

A recipe's `settings` reads top-down; omitted fields take the defaults in `settings.py`:

```json
{
  "steps": 300,
  "rollouts": {"prompts_per_step": 64, "samples_per_prompt": 8, "math_share": 1.0},
  "algorithm": {"name": "grpo", "policy": "on"},
  "optimizer": {"lr": 1e-6, "warmup_steps": 20},
  "replay": true, "replay_snapshot": "...", "replay_sources": {"...": 1.0},
  "replay_batch_size": 8, "replay_weight": 0.1
}
```

`uv run python -m trainer.rl --recipe RECIPE.json` prints the settings with every
default filled in, without Modal, data or a GPU.

**On- and off-policy.** `"policy": "on"` takes one optimizer update per rollout batch, so
every answer comes from the weights being updated. `"policy": "off"` with
`updates_per_batch: N` (at least 2) splits the batch's answers into N shares and takes
one update on each; after the first, answers come from older weights. The log-probabilities
of the weights that sampled the batch are computed before its first update, and the
algorithm corrects for the difference. Only algorithms listed in `OFF_POLICY` may run
off-policy.

**Algorithms.** An algorithm is a frozen dataclass registered in `ALGORITHMS`. It has
`advantages(rewards, group)` and `loss(tokens)`, where `tokens` is one micro-batch of
answers (`Tokens`: logits, log-probabilities now and when sampled, advantages). Its fields
are the recipe's `algorithm` options. `grpo` is the one implemented:

- Advantage: reward minus its group's mean. With `scale_by_std` it is also divided by
  the group's standard deviation; that is off by default, as in Dr. GRPO
  (arXiv:2503.20783).
- Loss: PPO's clipped surrogate −min(r·A, clip(r, 1 ± `clip`)·A) per answer token, with
  r = p / p_old: gradient −r·A·∇log p where the ratio is not clipped. On-policy r is 1,
  so the gradient is the plain policy gradient −A·∇log p (`tests/test_rl.py`).
- Score (`score`): `"plain"` pushes up ∇log p(y). `"centered"` uses Score Centering's
  ∇log p(y) − Σ_v q_v ∇log p_v ([`../trainer/rl/scores.py`](../trainer/rl/scores.py),
  the method behind Tinker's StableReinforce,
  [reference](../references/stable-reinforce-2026-10-25.md)). Here q is the sampler's
  distribution from its top `centering_top_k` (32) tokens, with the rest of the mass
  spread like the learner's. On the logits its gradient is onehot(y) − q instead of
  onehot(y) − p (`tests/test_rl_scores.py`). The sampler (cached one-token decoding) and
  the learner (a full forward pass) compute probabilities with different kernels;
  centering cancels any bias between them. Centering needs temperature 1.

## The loop

[`../trainer/rl/loop.py`](../trainer/rl/loop.py) on one H100:

1. **Prompts.** Draw `prompts_per_step` (64) training problems of the pool,
   `math_share` (half) math. A prompt solved by none or all of its group on two visits in
   a row rests for `rest_steps` (50) steps.
2. **Rollouts.** `samples_per_prompt` (8) answers per prompt at temperature 1, without
   top-k or top-p, so samples follow the policy's probabilities. At most 256 new tokens
   for math and 384 for code. The shared sampler
   [`../trainer/rollout.py`](../trainer/rollout.py) samples a left-padded batch with a
   key/value cache. The rollouts' default `engine`, `graph`, replays one CUDA graph per
   token built from custom decode kernels
   ([`../kernels/README.md`](../kernels/README.md#rollout-decoding)), 3.2 times faster
   than `eager` decoding on an H100 with the same error against the learner; the `eager`
   engine's greedy output matches unbatched, uncached decoding exactly
   (`tests/test_rollout.py`). It also records each sampled token's log-probability and,
   for a centered score, the sampler's top tokens.
3. **Rewards** ([`../data/rl_tasks.py`](../data/rl_tasks.py)). 1 or 0; an answer
   without EOS within the limit scores 0. Code answers are checked in the sandbox in a
   background thread while math is sampled.
4. **Advantages** from the algorithm. Groups whose answers all score the same have zero
   advantage and are skipped; their share is logged.
5. **Updates.** The algorithm's loss is summed over answer tokens and divided by the
   number of answers per update times the task's token limit (no per-answer length
   normalization).
6. **Anchors.** KL(policy ‖ parent) is estimated (k3) on the answer tokens every update and
   added with `kl_coef` (default 0); `sampler_kl_coef` (default 0) adds Tinker's
   sampler–learner KL term. Pretraining replay (`"replay": true`, the hook SFT
   also uses, [`../trainer/replay_hook.py`](../trainer/replay_hook.py)) adds a
   next-token loss on pretraining windows at `replay_weight` to every update.
7. **Optimizer.** AdamW without weight decay, LR 1e-6 after 20 warmup updates, gradient
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
answers trained on; optimizer updates so far; the algorithm's statistics (for `grpo`,
the share of clipped tokens); mean NLL of the sampled tokens; KL to the parent;
sampler–learner mismatch (mean sampler minus learner log-probability of the sampled
tokens); replay loss; gradient norm; resting prompts; generation and training seconds.
Every evaluation: validation accuracy and truncation per task, and replay validation
loss per source. With the graph engine at 64 prompts × 8 answers, a smoke run took 3.8 s
to generate and 0.9 s to train per step.

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
volume), `pool` (an `rl_pool` data ID) and `settings` ([code layout](#code-layout)). RL
checkpoints (format `rl-v1`) keep the model code and tokenizer beside them, so
`analysis/evaluation` loads them like SFT checkpoints.
