# RL for math and code

Reinforcement learning for our models (the subword 124M GPT and the hybrid 720M) on
problems whose answers can be checked automatically: math word problems, checked against the reference answer, and Python
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

The `s0` stage of [`../workflows/s0-rl-20b.json`](../workflows/s0-rl-20b.json) (run
`math-code-sft-replay-20b-s0-r02`): the 20B end point, 20,000 steps of 32 on
`mixture-math-code-02` (the 224,533 math rows and the Qwen3.5-solved code rows, [`../sft/CODE_DATA.md`](../sft/CODE_DATA.md)), r02's
cosine schedule and pretraining replay.

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

**Model families.** The parent checkpoint's recorded architecture picks the family
adapter ([`../trainer/families.py`](../trainer/families.py)); everything else in the
loop is shared:

| | Subword ([`../trainer/subword.py`](../trainer/subword.py) `SubwordModel`) | Hybrid ([`../hybrid/post_train.py`](../hybrid/post_train.py) `HybridModel`) |
| --- | --- | --- |
| Model, tokenizer | model_100m GPT, frozen subword tokenizer | hybrid LM (trunk, without MTP heads), pinned GLM tokenizer |
| Prompt | BOS and the task's SFT prompt (`rl_tasks.prompt`) | one user turn in GLM's chat template asking for a `\boxed{}` answer or one Python block |
| Answer ends at | EOS | EOS, `<\|user\|>` or `<\|observation\|>` |
| Answer format | `plain`: the whole completion | `boxed`: the last `\boxed{}`; the last fenced Python block |
| Sampler | [`../trainer/rollout.py`](../trainer/rollout.py) (records log-probabilities and top tokens) | [`../hybrid/generate.py`](../hybrid/generate.py) on a bf16 copy of the policy (no log-probabilities, so no StableReinforce) |
| Replay, exports | pretraining replay; the checkpoint is the artifact | no replay; a bf16 export at the end, uploaded to `hf_repo` if set |

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

**StableReinforce** (`"stable_reinforce": {"top_k": 32, "sampler_kl_coef": 0.0}`, off
when absent) is an off-policy correction that any algorithm takes on answers our sampler
wrote: [`../trainer/rl/off_policy/stable_reinforce.py`](../trainer/rl/off_policy/stable_reinforce.py),
the method behind Tinker's StableReinforce
([reference](../references/stable-reinforce-2026-10-25.md)). The sampler (cached
one-token decoding) and the learner (a full forward pass) compute probabilities with
different kernels, so even fresh samples are slightly off-policy. The hook replaces each
token's score ∇log p(y), which the algorithm's loss pushes up, by Score Centering's
∇log p(y) − Σ_v q_v ∇log p_v. q is the sampler's distribution from its top `top_k`
tokens, with the rest of the mass spread like the learner's; on the logits the gradient
is onehot(y) − q instead of onehot(y) − p, so any bias between the engines cancels
(`tests/test_stable_reinforce.py`). `sampler_kl_coef` adds Tinker's sampler–learner KL
term. Centering needs temperature 1.
`tests/test_rl.py` checks that GRPO's gradient is the plain policy gradient, PPO's
clipping, and AWR's weights.

## The loop

[`../trainer/rl/loop.py`](../trainer/rl/loop.py), on the stage's `gpus` H100s (see
[24-hour limit](#24-hour-limit)). Steps 1–3 are the `sampler` and `reuse` sources; the `file` source draws
`answers_per_step` rewarded answers instead.

1. **Prompts.** Draw `prompts_per_step` (64) training problems of the pool,
   `math_share` (half) math. A prompt solved by none or all of its group on two visits in
   a row rests for `rest_steps` (50) steps.
2. **Rollouts.** `samples_per_prompt` (8) answers per prompt at temperature 1, without
   top-k or top-p, so samples follow the policy's probabilities. At most 256 new tokens
   for math and 384 for code. The family's sampler draws them; for subword, the shared sampler
   [`../trainer/rollout.py`](../trainer/rollout.py) samples a left-padded batch with a
   key/value cache. The rollouts' default `engine`, `graph`, replays one CUDA graph per
   token built from custom decode kernels
   ([`../kernels/README.md`](../kernels/README.md#rollout-decoding)), 3.2 times faster
   than `eager` decoding on an H100 with the same error against the learner; the `eager`
   engine's greedy output matches unbatched, uncached decoding exactly
   (`tests/test_rollout.py`). It also records each sampled token's log-probability and,
   for StableReinforce, the sampler's top tokens.
3. **Rewards** ([`../data/rl_tasks.py`](../data/rl_tasks.py)). 1 or 0, reading the
   answer in the family's format; an answer that does not end in one of the family's
   stop tokens within the limit scores 0. Code answers are checked in the sandbox in a
   background thread while math is sampled; a pool row's tests run as `functions` (every
   `test*` function is called) or, for KodCode, under `pytest`.
4. **Advantages** from the algorithm. For GRPO and PPO, groups whose answers all score
   the same have zero advantage and are skipped; their share is logged.
5. **Updates.** The algorithm's loss is summed over answer tokens and divided by the
   number of answers per update times the task's token limit (no per-answer length
   normalization).
6. **Anchors.** KL(policy ‖ parent) is estimated (k3) on the answer tokens every update and
   added with `kl_coef` (default 0). Pretraining replay (`"replay": true`, the hook SFT
   also uses, [`../trainer/replay_hook.py`](../trainer/replay_hook.py)) adds a
   next-token loss on pretraining windows at `replay_weight` to every update.
7. **Optimizer.** The role-declared optimizer ([`../trainer/optimizers.py`](../trainer/optimizers.py)):
   by default AdamW for every parameter without weight decay, LR 1e-6 after 20 warmup
   updates; with `"muon": true` Muon at `muon_lr` for hidden matrices. Gradient clipping
   at 1.0.

Validation is greedy accuracy on the pool's validation problems plus replay validation
loss, at the start, every `evaluate_every` steps and at the end.

### 24-hour limit

RL runs as a stage of a [workflow](../workflows/README.md), under the launch's deadline
and supervisor: one training call of at most 21 hours (`max_seconds` at most 20), no
retries. Training stops taking updates 30 minutes before the deadline. The code sandbox
is recorded for the supervisor, which terminates it when the launch ends. A run saves a
checkpoint every `save_every` steps, every `save_seconds` (an hour) and when it stops,
with its optimizer, prompt scheduler, trajectory source and random generators; the
last `keep_checkpoints` are kept. A run that stops before its step target (the time
limit, the deadline, or the workflow's `stop` entrypoint) leaves the stage paused, and
the next launch of the workflow run resumes it exactly in the same run folder; only
the step target and the save, validation and time settings may change. The final
validation and the family's export follow the step target.

The stage's `gpus` (1–8) sets the H100s ([`../trainer/rl/ranks.py`](../trainer/rl/ranks.py)):
one process per GPU, every process drawing the same prompts and sampling and training
its share (`prompts_per_step` must split evenly), with gradients averaged before
clipping. Validation is sharded the same way. The launching process holds the code
sandbox and serves it, volume commits and the stop request to the processes.

### Logged

Every step: reward, share of uniform groups, answer length and truncation rate per task;
answers trained on; optimizer updates so far; the algorithm's statistics (for `grpo`,
the share of clipped tokens); mean NLL of the sampled tokens; KL to the parent; with
StableReinforce, the sampler–learner mismatch (mean sampler minus learner
log-probability of the sampled tokens); replay loss; gradient norm; resting prompts; generation and training seconds.
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

## Runs

The RL branches of [`../workflows/s0-rl-20b.json`](../workflows/s0-rl-20b.json), all from
the stage 0 checkpoint on `rl-pool-math-code-03`, with replay at weight 0.1:

| Run | Purpose | Success looks like |
| --- | --- | --- |
| S0 | Joint math + code SFT with replay | Code pass@1 above 0, math and general scores at r02 level |
| R1 (`r1-math`) | Math-only RL, 300 steps | GSM8K rises; validates the loop cheaply |
| R2 (`r2-code`) | Code-only RL, 300 steps | MBPP+ pass@1 rises; validates the sandbox reward |
| R3 (`r3-joint`) | Joint RL, 1,000 steps | Both rise; general scores within the targets |

## Results (2026-10-26)

R1–R3 ran with the RL trainer before it was split into `trainer/rl/` (commit `ef2dc8a`):
on-policy GRPO with a centered score (top 32), the graph engine and replay windows
starting at random documents. The workflow states that algorithm as
`"algorithm": {"name": "on_policy/grpo"}` with `"stable_reinforce": {"top_k": 32}`.
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

RL is a workflow stage ([`../workflows/stages/rl.py`](../workflows/stages/rl.py)). Its
entry holds `pool` (an `rl_pool` data ID, also used for validation) and `settings`
([code layout](#code-layout)); the workflow supplies its parent, usually an SFT stage
or a `from` checkpoint. [`../workflows/s0-rl-20b.json`](../workflows/s0-rl-20b.json)
holds stage 0 (`s0`) and then R1, R2 and R3 (`r1-math`, `r2-code`, `r3-joint`) as
branches from it. [`../workflows/hybrid-720m-rl.json`](../workflows/hybrid-720m-rl.json)
is the hybrid run: chat SFT of the 40B export as a `midtrain` stage, then RL on a pool
of GSM8K, MATH levels 1–3 and KodCode-Light-RL easy/medium problems (the old `rl-v1`
mix; KodCode kept only where the reference solution passes in the sandbox) with Muon,
16 prompts × 8 answers per step and 768-token answers.

From `training/`, after the data of [`../sft/CODE_DATA.md`](../sft/CODE_DATA.md):

```sh
# Every setting filled in, no Modal.
uv run python -m workflows --workflow workflows/s0-rl-20b.json --run-id ID
# Stage 0, then R1-R3 from it.
uv run --all-extras modal run --detach -m workflows.modal_train::main \
  --workflow workflows/s0-rl-20b.json --run-id ID
# Only RL, from the existing stage 0 checkpoint.
uv run --all-extras modal run --detach -m workflows.modal_train::main \
  --workflow workflows/s0-rl-20b.json --run-id ID --start r1-math \
  --checkpoint math-code-sft-replay-20b-s0-r02/step-000020000.pt
# The hybrid pool, then chat SFT and RL.
uv run --all-extras modal run -m modal_scripts.modal_sft_data::rl_pool \
  --data-id rl-pool-hybrid-02 --math-lighteval --kodcode --code-train 0
uv run --all-extras modal run --detach -m workflows.modal_train::main \
  --workflow workflows/hybrid-720m-rl.json --run-id ID
# Off-policy trajectories from the Qwen teacher's saved code answers:
uv run --all-extras modal run -m modal_scripts.modal_teacher::trajectories \
  --teacher code-teacher-qwen35-02 --data-id <fresh-id> [--limit N]
```

To play with the stage 0 and RL checkpoints next to the base and math SFT models, export
them to the private comparison Space ([`../analysis/playground`](../analysis/playground/README.md)):

```sh
uv run --all-extras modal run -m playground.modal_export \
  --config analysis/playground/configs/post-training.json --name EXPORT-NAME
```

RL checkpoints (format `rl-v1`, `"stage": "rl"`) record their architecture and keep the
family's source files and tokenizer beside them, so evaluation and later stages load
them like any other checkpoint (`trainer/checkpoints.py`). A parent must record its
architecture: pretraining checkpoints and SFT/RL checkpoints from this layout on.
