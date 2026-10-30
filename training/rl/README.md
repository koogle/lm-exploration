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
algorithm or a source of answers can change without touching the rest:

| Path | Owns |
| --- | --- |
| `settings.py` | What a recipe sets, in sections: `algorithm`, `trajectories`, `rollouts`, `optimizer`, plus the shared `replay_*` fields |
| `algorithms.py` | The algorithms by name |
| `on_policy/` | Algorithms that train only on answers the current weights just sampled: `grpo.py` |
| `off_policy/` | Algorithms that also train on older or other models' answers (`ppo.py`, `awr.py`), and their trajectory sources (`sources.py`) |
| `rollouts.py` | Sampling answers with the current weights and rewarding them |
| `batch.py`, `policy_gradient.py` | What the algorithms share |
| `loop.py` | The training loop, checkpoints and provenance |

A recipe's `settings` reads top-down; omitted fields take the defaults in `settings.py`:

```json
{
  "steps": 300,
  "algorithm": {"name": "on_policy/grpo"},
  "rollouts": {"prompts_per_step": 64, "samples_per_prompt": 8, "math_share": 1.0},
  "optimizer": {"lr": 1e-6, "warmup_steps": 20},
  "replay": true, "replay_snapshot": "...", "replay_sources": {"...": 1.0},
  "replay_batch_size": 8, "replay_weight": 0.1
}
```

**Where answers come from** (`trajectories.source`):

| Source | Answers | Updates per batch | Behaviour log-probabilities |
| --- | --- | --- | --- |
| `sampler` (default) | Sampled by the current weights every step | 1 | The current weights' own |
| `reuse` | Sampled the same way, reused | `updates_per_batch` (≥ 2), each on a share | Computed before the batch's first update |
| `file` | A trajectory file (`data`), `answers_per_step` per step | 1 | None |

A trajectory file ([`../data/rl_trajectories.py`](../data/rl_trajectories.py),
`rl-trajectories-v1`) holds one answer per row: its problem, the answer text and its
reward. `modal_scripts/modal_teacher.py::trajectories` makes one from the Qwen teacher's
saved code answers, passing and failing. The teacher uses another tokenizer, so
its answers have no log-probabilities our model could use for importance ratios.

**Algorithms** (`algorithm.name`; its other fields are the algorithm's options):

| Name | Sources | Advantage | Loss per answer token |
| --- | --- | --- | --- |
| `on_policy/grpo` | `sampler` | Reward minus the group mean; with `scale_by_std`, divided by the group's standard deviation (off by default, as in Dr. GRPO, arXiv:2503.20783) | −A·∇log p |
| `off_policy/ppo` | `reuse` | As GRPO | PPO's clipped surrogate −min(r·A, clip(r, 1 ± `clip`)·A), r = p / p_old |
| `off_policy/awr` | `reuse`, `file` | Reward minus the batch mean of its task | −w·∇log p with w = min(exp(A / `beta`), `max_weight`) (advantage-weighted regression, arXiv:1910.00177) |

An algorithm is a frozen dataclass registered in `algorithms.py`. It declares the sources
it accepts and whether it needs behaviour log-probabilities, and has
`advantages(batch)` and `loss(tokens)`. `tokens` is one micro-batch of answers
(`Tokens`: logits, log-probabilities now and when sampled, advantages). A recipe pairing
an algorithm with a source it does not accept fails when its settings are checked.
Policy-gradient algorithms skip answers with zero advantage; AWR trains every answer.
`tests/test_rl.py` checks that GRPO's gradient is the plain policy gradient, PPO's
clipping, and AWR's weights.

## The loop

[`../trainer/rl/loop.py`](../trainer/rl/loop.py) on one H100. Steps 1–3 are the `sampler`
and `reuse` sources; the `file` source draws `answers_per_step` rewarded answers instead.

1. **Prompts.** Draw `prompts_per_step` (64) training problems of the pool,
   `math_share` (half) math. A prompt solved by none or all of its group on two visits in
   a row rests for `rest_steps` (50) steps.
2. **Rollouts.** `samples_per_prompt` (8) answers per prompt at temperature 1, without
   top-k or top-p, so samples follow the policy's probabilities. At most 256 new tokens
   for math and 384 for code. The shared sampler
   [`../trainer/rollout.py`](../trainer/rollout.py) samples a left-padded batch with a
   key/value cache; its greedy output matches unbatched, uncached decoding exactly
   (`tests/test_rollout.py`).
3. **Rewards** ([`../data/rl_tasks.py`](../data/rl_tasks.py)). 1 or 0; an answer
   without EOS within the limit scores 0. Code answers are checked in the sandbox in a
   background thread while math is sampled.
4. **Advantages** from the algorithm. For GRPO and PPO, groups whose answers all score
   the same have zero advantage and are skipped; their share is logged.
5. **Updates.** The algorithm's loss is summed over answer tokens and divided by the
   number of answers per update times the task's token limit (no per-answer length
   normalization).
6. **Anchors.** KL(policy ‖ parent) is estimated (k3) on the answer tokens every update and
   added with `kl_coef` (default 0). Pretraining replay (`"replay": true`, the hook SFT
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
answers trained on; optimizer updates so far; the algorithm's statistics (for `grpo`, the share of
clipped tokens); mean NLL of the sampled tokens; KL to the parent; replay loss; gradient
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
volume), `pool` (an `rl_pool` data ID) and `settings` ([code layout](#code-layout)). RL
checkpoints (format `rl-v1`) keep the model code and tokenizer beside them, so
`analysis/evaluation` loads them like SFT checkpoints.
