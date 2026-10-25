# RL for math and code

Reinforcement learning for our 124M models on problems whose answers can be checked
automatically: math word problems, checked against the reference answer, and Python
functions, checked by running unit tests. The model writes complete answers and is
trained on its own samples, scored by those checkers. In SFT
([`../sft/README.md`](../sft/README.md)) it only imitates answers someone else wrote.

**Status:** implemented and smoke-tested on Modal (4 full-size steps from the r02 math
model). The stage 0 teacher data is being generated; no RL run has been measured yet.

## Starting point

Measured so far ([`../sft/MATH_EXPERIMENTS.md`](../sft/MATH_EXPERIMENTS.md)): the 20B base
model scores about 2% on GSM8K. Math SFT r01 reaches 20.4 but forgets general ability;
math SFT with pretraining replay (r02) reaches 17.4 on GSM8K and 16.7 on SVAMP and keeps
most of it. No model had been trained to write a function for a problem statement.

RL can only reinforce answers the model already produces sometimes: a prompt the model
never solves gives no learning signal. So code needs an SFT stage before RL.

## Targets

| Measure | Now (best SFT) | Target after RL |
| --- | --- | --- |
| GSM8K (500 test, greedy, Math-Verify) | 17.4 | ≥ 30 |
| SVAMP (300, greedy) | 16.7 | ≥ 25 |
| MBPP+ / HumanEval+ greedy pass@1 | stage 0 pending | clearly above stage 0 |
| ClimbMix bits per byte | 0.890 (base 0.876) | within 3% of base |
| HellaSwag, ARC-E | 38.2, 53.1 | within 2 points of the starting checkpoint |

These are aspirations for choosing settings, not predictions. For scale: GPT-3 175B
fine-tuned on GSM8K reached about 33%.

## Stage 0: joint SFT (math + code + replay)

`trainer/sft.py` with `"task": "mixture"` ([`../data/mixture_sft.py`](../data/mixture_sft.py)):
each row is a math row ([`math_sft.py`](../data/math_sft.py)) or a code row
([`code_sft.py`](../data/code_sft.py)) and keeps its task's prompt and answer. Code rows:

- **Prompt:** ``Question: {problem}\nWrite a Python function `def f(args):`.\nAnswer:\n``
  (or ``class `class C:` ``); **answer:** the complete solution, then EOS.
- **Problems** (`code-oci-mbpp-01`, `modal_sft_data::code`): 227,084 NVIDIA
  OpenCodeInstruct problems (CC-BY-4.0) from six of its 50 shards, each with ten unit
  tests, whose own Qwen2.5-Coder-32B solution passed all ten, plus 306 MBPP train
  problems; 453 are validation. Rows sharing a 13-word n-gram with an MBPP
  test/validation/prompt, MBPP+ or HumanEval+ problem are dropped (80).
- **Solutions** ([`../data/teacher.py`](../data/teacher.py),
  [`../modal_scripts/modal_teacher.py`](../modal_scripts/modal_teacher.py)): Qwen3.5-35B-A3B
  (Apache-2.0, FP8, thinking off, vLLM) answers each problem twice; every answer runs
  against the problem's tests in the code sandbox, and the shortest passing answer
  becomes the row's solution. A 64-problem trial kept 59 problems, with a median
  solution of 311 characters against 593 for the OpenCodeInstruct solutions.
- **Licences:** problems and tests CC-BY-4.0 (OpenCodeInstruct), MBPP CC-BY-4.0, teacher
  Apache-2.0, so trained weights can be published with attribution.
- **Mixture** (`mixture-math-code-02`): the 224,533 math rows and the kept code rows,
  sampled uniformly.
- **Run** ([`../sft/recipes/math-code-replay-20b.json`](../sft/recipes/math-code-replay-20b.json)):
  the 20B end point, 20,000 steps of 32, r02's cosine schedule and replay.

## Stage 1: the RL loop

[`../trainer/rl.py`](../trainer/rl.py) on one H100; the policy samples with the current
weights, so the loop is fully on-policy. Each step:

1. **Prompts.** Draw `prompts_per_step` (64) training problems of the pool,
   `math_share` (half) math. A prompt solved by none or all of its group on two visits in
   a row rests for `rest_steps` (50) steps.
2. **Rollouts.** `samples_per_prompt` (8) answers per prompt at temperature 1, without
   top-k or top-p, so samples follow the policy's probabilities: at most 256 new tokens
   for math and 384 for code. [`../trainer/rollout.py`](../trainer/rollout.py) samples a
   left-padded batch with a key/value cache. The default `graph` engine replays one
   CUDA graph per token built from custom decode kernels
   ([`../kernels/README.md`](../kernels/README.md#rollout-decoding)), 3.2 times faster
   than eager decoding on an H100; the `eager` engine's greedy output matches unbatched,
   uncached decoding exactly (`tests/test_rollout.py`). The sampler also records each
   token's log-probability and its top-32 tokens.
3. **Rewards** ([`../data/rl_tasks.py`](../data/rl_tasks.py)). 1 or 0; an answer
   without EOS within the limit scores 0. Math: Math-Verify finds the final answer equal
   to the reference. Code: every test passes in the sandbox (below). Code answers are
   checked in a background thread while math is sampled.
4. **Advantages.** Reward minus the mean reward of the prompt's group, with no division
   by the group's standard deviation (Dr. GRPO, arXiv:2503.20783). Groups whose answers
   all score the same have zero advantage and are skipped; their share is logged.
5. **Loss.** −advantage × score summed over answer tokens, divided by the number of
   answers times the task's token limit (no per-answer length normalization). One
   optimizer step per batch, so the loop is on-policy and needs no importance ratio or
   clipping. The score is Score Centering's (`score_centering_top_k`, default 32;
   [`../trainer/policy_loss.py`](../trainer/policy_loss.py), the method behind Tinker's
   StableReinforce, [reference](../references/stable-reinforce-2026-10-25.md)):
   ∇log p(y) − Σ_v q_v ∇log p_v, where q is the sampler's distribution from its top-32
   with the rest of the mass spread like the learner's. Its gradient on the logits is
   onehot(y) − q instead of REINFORCE's onehot(y) − p, so a bias between the sampler's
   numerics and the learner's cancels (`tests/test_policy_loss.py`). That bias is
   logged every step as `sampler_mismatch`: in the smoke runs it was about 1e-4 nats per
   token on average, with about 0.015 mean absolute difference per token.
6. **Anchors.** KL(policy ‖ parent) is estimated (k3) on the answer tokens every step and
   added with `kl_coef` (default 0); `sampler_kl_coef` (default 0) adds Tinker's
   sampler–learner KL term. Pretraining replay (`replay_*`, as in SFT) adds a
   next-token loss on pretraining windows at `replay_weight`.
7. **Optimizer.** AdamW without weight decay, LR 1e-6 after 20 warmup steps, gradient
   clipping at 1.0. 1e-6 is a starting guess.

Validation is greedy accuracy on the pool's validation problems (199 of the first
200 GSM8K training problems, which SFT only validates on, and 77 MBPP validation problems) plus
replay validation loss, at the start, every `evaluate_every` steps and at the end.

### RL pool

`rl-pool-math-code-03` (`modal_sft_data::rl_pool`): 6,971 GSM8K training problems after
the first 500 (SFT's validation problems), 10,000 OpenCodeInstruct problems from its last
shard (code SFT uses the first six), and the validation sets above; decontaminated like the SFT data. Prompts longer than 512
tokens are not used.

### Code execution sandbox

Model-written code runs only in a Modal Sandbox
([`../data/code_execution.py`](../data/code_execution.py)): `block_network=True`, CPU
only, no volumes, secrets or HF token, and a `timeout` ending at the launch deadline.
[`../data/code_runner.py`](../data/code_runner.py) runs each program as `python -I -`
in a fresh directory with 5 s of CPU, 1 GB of memory and a 16 MB file-size limit. The
program imports the solution as the module `solution`, runs the tests, calls every
`test*` function they define and then prints a random per-program nonce; it passes only
with exit code 0 and the nonce as its last line, so a solution that exits early or skips
the tests fails.

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
At 64 prompts × 8 answers the smoke run took 3.8 s to generate and 0.9 s to train per
step. Every evaluation: validation
accuracy and truncation per task, and replay validation loss per source.

Signs to watch: rewards rising while EvalPlus pass@1 does not (reward hacking), sampled
NLL collapsing toward zero while reward plateaus (lost exploration), truncation rising.

## Evaluation

[`analysis/evaluation`](../analysis/evaluation/README.md) adds MBPP+ (378) and
HumanEval+ (164) greedy pass@1 in the same sandbox, next to GSM8K, SVAMP, the general
benchmarks and held-out bits per byte.

## Runs

| Run | Purpose | Success looks like |
| --- | --- | --- |
| S0 | Joint math + code SFT with replay | Code pass@1 above 0, math and general scores at r02 level |
| R1 | Math-only RL from S0, 300 steps | GSM8K rises; validates the loop cheaply |
| R2 | Code-only RL from S0, 300 steps | MBPP+ pass@1 rises; validates the sandbox |
| R3 | Joint RL from S0, 1,000 steps | Both rise; general scores within the targets |
| Ablations | 8 vs 16 samples; replay on or off; LR 5e-7 / 1e-6 / 3e-6 | Settings for R3 |

## Run

From `training/`:

```sh
# Data (once): code SFT rows, the math + code mixture and the RL pool.
uv run --all-extras modal run -m modal_scripts.modal_sft_data::code --data-id code-oci-mbpp-01
uv run --all-extras modal run -m modal_scripts.modal_teacher --source code-oci-mbpp-01 \
  --data-id code-teacher-qwen35-01 --shards 8 --samples 2
uv run --all-extras modal run -m modal_scripts.modal_sft_data::mixture --data-id mixture-math-code-02 \
  --math math-gsm8k-metamath-02 --code code-teacher-qwen35-01
uv run --all-extras modal run -m modal_scripts.modal_sft_data::rl_pool --data-id rl-pool-math-code-03
# Stage 0 SFT, then RL from its final checkpoint.
uv run --all-extras modal run -m modal_scripts.modal_sft_train::main --recipe sft/recipes/math-code-replay-20b.json
uv run --all-extras modal run -m modal_scripts.modal_rl_train --recipe rl/recipes/RECIPE.json
```

An RL recipe names `run_id`, `parent` (`run/step-N.pt`), `pool` (a pool data ID) and
`settings` (`trainer.rl.Settings`). RL checkpoints (format `rl-v1`) keep the model code
and tokenizer beside them, so `analysis/evaluation` loads them like SFT checkpoints.
