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
| `settings.py` | What an RL stage's `settings` set, in sections: `algorithm`, `trajectories`, `rollouts`, `optimizer`, plus the shared `replay_*` fields |
| `algorithms.py` | The algorithms by name |
| `on_policy/` | Algorithms that train only on answers the current weights just sampled: `grpo.py` |
| `off_policy/` | Algorithms that also train on older or other models' answers (`ppo.py`, `awr.py`), and their trajectory sources (`sources.py`) |
| `rollouts.py` | Sampling answers with the current weights and rewarding them |
| `batch.py`, `policy_gradient.py` | What the algorithms share |
| `loop.py` | The training loop, checkpoints and provenance |

A stage's `settings` read top-down; omitted fields take the defaults in `settings.py`:

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
(`Tokens`: logits, log-probabilities now and when sampled, advantages). Settings that pair
an algorithm with a source it does not accept fail when they are checked.
Policy-gradient algorithms skip answers with zero advantage; AWR trains every answer.

**Score.** GRPO and PPO take `score`: `"plain"` pushes up ∇log p(y), `"centered"` uses
Score Centering's ∇log p(y) − Σ_v q_v ∇log p_v
([`../trainer/rl/scores.py`](../trainer/rl/scores.py), the method behind Tinker's
StableReinforce, [reference](../references/stable-reinforce-2026-10-25.md)). q is the
sampler's distribution from its top `centering_top_k` (32) tokens, with the rest of the
mass spread like the learner's; on the logits the gradient is onehot(y) − q instead of
onehot(y) − p (`tests/test_rl_scores.py`). The sampler (cached one-token decoding) and
the learner (a full forward pass) compute probabilities with different kernels;
centering cancels any bias between them. It needs temperature 1 and answers our sampler
wrote.
`tests/test_rl.py` checks that GRPO's gradient is the plain policy gradient, PPO's
clipping, and AWR's weights.

## The loop

[`../trainer/rl/loop.py`](../trainer/rl/loop.py); the workflow stage currently runs it on
one H100. Steps 1–3 are the `sampler` and `reuse` sources; the `file` source draws
`answers_per_step` rewarded answers instead.

1. **Prompts.** Draw `prompts_per_step` (64) training problems of the pool,
   `math_share` (half) math. A prompt solved by none or all of its group on two visits in
   a row rests for `rest_steps` (50) steps.
2. **Rollouts.** `samples_per_prompt` (8) answers per prompt at temperature 1, without
   top-k or top-p, so samples follow the policy's probabilities. At most 256 new tokens
   for math and 384 for code. The shared sampler
   [`../trainer/rollout.py`](../trainer/rollout.py) samples a left-padded batch with a
   key/value cache; its greedy output matches unbatched, uncached decoding exactly
   (`tests/test_rollout.py`). It also records each sampled token's log-probability and,
   for a centered score, the sampler's top tokens.
3. **Rewards** ([`../data/rl_tasks.py`](../data/rl_tasks.py)). 1 or 0; an answer
   without EOS within the limit scores 0. Code answers are checked in the sandbox in a
   background thread while math is sampled.
4. **Advantages** from the algorithm. For GRPO and PPO, groups whose answers all score
   the same have zero advantage and are skipped; their share is logged.
5. **Updates.** The algorithm's loss is summed over answer tokens and divided by the
   number of answers per update times the task's token limit (no per-answer length
   normalization).
6. **Anchors.** KL(policy ‖ parent) is estimated (k3) on the answer tokens every update and
   added with `kl_coef` (default 0); `sampler_kl_coef` (default 0) adds Tinker's
   sampler–learner KL term on answers our sampler wrote. Pretraining replay (`"replay": true`, the hook SFT
   also uses, [`../trainer/replay_hook.py`](../trainer/replay_hook.py)) adds a
   next-token loss on pretraining windows at `replay_weight` to every update.
7. **Optimizer.** AdamW without weight decay, LR 1e-6 after 20 warmup updates, gradient
   clipping at 1.0.

Validation is greedy accuracy on the pool's validation problems plus replay validation
loss, at the start, every `evaluate_every` steps and at the end.

### 24-hour limit

RL runs as a stage of a [workflow](../workflows/README.md), under the launch's deadline
and supervisor: one training call of at most 21 hours (`max_seconds` at most 20), no
retries. Training stops taking updates 30 minutes before the deadline. The code sandbox
is recorded for the supervisor, which terminates it when the launch ends. A run does
not resume: a relaunch reruns an unfinished RL stage from its parent.

### Logged

Every step: reward, share of uniform groups, answer length and truncation rate per task;
answers trained on; optimizer updates so far; the algorithm's statistics (for `grpo`,
the share of clipped tokens); mean NLL of the sampled tokens; KL to the parent;
sampler–learner mismatch (mean sampler minus learner log-probability of our sampler's
tokens); replay loss; gradient norm; resting prompts; generation and training seconds.
Every evaluation: validation accuracy and truncation per task, and replay validation
loss per source.

Signs to watch: rewards rising while EvalPlus pass@1 does not (reward hacking), sampled
NLL collapsing toward zero while reward plateaus (lost exploration), truncation rising.

## Evaluation

[`analysis/evaluation`](../analysis/evaluation/README.md) adds MBPP+ (378) and
HumanEval+ (164) greedy pass@1 in the code sandbox, next to GSM8K, SVAMP, the general
benchmarks and held-out bits per byte. Reference solutions of a 40-problem sample of
each pass all their tests through this harness.

## Run

RL is a workflow stage ([`../workflows/stages/rl.py`](../workflows/stages/rl.py)). Its
entry holds `pool` (an `rl_pool` data ID, also used for validation) and `settings`
([code layout](#code-layout)); the workflow supplies its parent, usually an SFT stage
or a `from` checkpoint. A workflow can hold SFT then RL, or only RL stages from a
checkpoint, branching into several algorithms:

```json
{
  "name": "code-rl",
  "family": "subword",
  "from": "<sft-run>/step-NNNNNNNNN.pt",
  "stages": [
    {"stage": "rl", "branches": [
      {"name": "grpo", "pool": "<pool-id>", "settings": {"algorithm": {"name": "on_policy/grpo"}}},
      {"name": "awr", "pool": "<pool-id>", "settings": {
        "algorithm": {"name": "off_policy/awr"},
        "trajectories": {"source": "file", "data": "<trajectories-id>"}}}
    ]}
  ]
}
```

From `training/`:

```sh
uv run python -m workflows --workflow WORKFLOW.json --run-id ID   # resolved, no Modal
uv run --all-extras modal run --detach -m workflows.modal_train::main \
  --workflow WORKFLOW.json --run-id ID [--start STAGE] [--checkpoint RUN/step-N.pt]
# Off-policy trajectories from the Qwen teacher's saved code answers:
uv run --all-extras modal run -m modal_scripts.modal_teacher::trajectories \
  --teacher code-teacher-qwen35-02 --data-id <fresh-id> [--limit N]
```

RL checkpoints (format `rl-v1`, `"stage": "rl"`) keep the model code and tokenizer
beside them, so evaluation and later stages load them like any other checkpoint
(`trainer/checkpoints.py`).
