# RL for math and code (design)

**Status: design only. Nothing in this document is implemented or measured yet.**

This is a reinforcement-learning loop that trains our 124M models on problems whose
answers can be checked automatically: math word problems, checked against the reference
answer, and Python functions, checked by running unit tests. The model writes complete
answers and is trained on its own samples, scored by those checkers. In SFT
([`../sft/README.md`](../sft/README.md)) it only imitates answers someone else wrote.

## Starting point

Measured so far ([`../sft/MATH_EXPERIMENTS.md`](../sft/MATH_EXPERIMENTS.md)):

- 20B base model: about 2% on GSM8K and 1% on SVAMP.
- Math SFT r01: 20.4 on GSM8K, but it forgot much of its general ability (ClimbMix bits
  per byte +35%, ARC-E −11 points).
- Math SFT with pretraining replay, r02 (20B parent): 17.4 on GSM8K and 16.7 on SVAMP,
  scored with Math-Verify, while keeping most general ability.
- No model has seen an instruction-to-function coding task, and no code benchmark is
  wired into [`../analysis/evaluation`](../analysis/evaluation/README.md) yet.

RL can only reinforce answers the model already produces sometimes. A prompt the model
never solves in a group of samples gives no learning signal. So code needs an SFT stage
before RL, and both tasks need a pool of prompts the model solves only some of the time.

## Targets

| Measure | Now (best SFT) | Target after RL |
| --- | --- | --- |
| GSM8K (500 test, greedy, Math-Verify) | 17.4 | ≥ 30 |
| SVAMP (300, greedy) | 16.7 | ≥ 25 |
| MBPP+ / HumanEval+ pass@1 | not measured | clearly above the stage 0 SFT model |
| ClimbMix bits per byte | 0.890 (base 0.876) | within 3% of base |
| HellaSwag, ARC-E | 38.2, 53.1 | within 2 points of the starting checkpoint |

These are aspirations for choosing settings, not predictions. For scale: GPT-3 175B
fine-tuned on GSM8K reached about 33%.

## Stage 0: joint SFT (math + code + replay)

This reuses `trainer/sft.py` as it is, with a new `code` task in `data/` that subclasses
`SFTDataset`:

- **Prompt:** the problem statement and the function signature.
- **Answer:** the complete reference solution.
- **Data:** KodCode-V1 (`KodCode/KodCode-V1`, pinned revision; synthetic Python
  problems, each with a solution and pytest tests), plus the MBPP train split.
  - KodCode is filtered to solutions of at most 512 tokens and its easier subsets.
  - The tests are kept for RL.
- **Licence:** KodCode is CC-BY-NC-4.0. That's fine for this private research, but it
  restricts publishing derived weights.
- **Decontamination:** the 13-gram filter from `data/math_sft.py` removes prompts that
  overlap HumanEval, MBPP or EvalPlus test problems.
- **Run settings:** a single run mixes the math and code tasks (sampling the two
  datasets by weight) and keeps the r02 replay and cosine schedule.
- **Parent:** the 20B end point, or whichever r02 checkpoint scores best.

Gate before RL: on 512 held-out training-pool prompts per task, sampled 8 times each
at temperature 1, at least 25% of prompts should have a pass rate strictly between 0
and 1. Otherwise more or easier SFT comes first.

## Stage 1: the RL loop

One H100 runs everything except code execution. The policy samples with the current
weights, so the loop is fully on-policy.

Each step:

1. **Prompts.** Draw 64 prompts, half math (GSM8K and MetaMathQA training questions,
   decontaminated as in `data/math_sft.py`) and half code (the KodCode/MBPP pool from
   stage 0). Skip prompts whose recent pass rate is 0/8 or 8/8 on two visits in a row;
   they come back after N steps.
2. **Rollouts.** Sample G = 8 answers per prompt (512 sequences) at temperature 1.0,
   without top-k or top-p, so the sampling matches the policy's probabilities. Limits:
   256 new tokens for math, 384 for code. This needs a batched KV-cache sampler on the
   GPU (`trainer/rollout.py`); the playground's `generation.py` samples one sequence at
   a time on the CPU.
3. **Rewards.** Binary. An answer that does not end in EOS within the limit scores 0.
   - Math: 1 if Math-Verify finds the `#### N` answer equal to the reference.
   - Code: 1 if every test passes in the sandbox (below).
4. **Advantages.** Group-relative: for each sample, its reward minus its group's mean
   reward.
   - No division by the group's standard deviation, and the loss is not normalized per
     sequence length. Dr. GRPO (arXiv:2503.20783) shows both bias training toward
     longer wrong answers.
   - Groups with identical rewards have zero advantage. Their share is logged, because
     it measures how many prompts are wasted.
5. **Loss.** Policy gradient −A · log p(token) summed over the answer tokens of all
   samples, divided by a constant (512 × the token limit). Prompt tokens carry no loss.
   - One optimizer step per batch: the samples come from the current weights, so no
     importance ratio or clipping is needed.
   - PPO-style clipping is added only if a later variant reuses a batch for several
     steps.
6. **Anchor.** Add the r02 replay term: 8 pretraining windows per step, at a small
   weight (start at 0.1). Log the KL divergence to the starting checkpoint on the
   rollout tokens, but don't penalize it at first. If general benchmarks slip or the
   KL keeps growing, add a KL penalty.
7. **Optimizer.** AdamW, constant LR 1e-6 after 20 warmup steps, gradient clipping at
   1.0. 1e-6 is a starting guess; sweep 5e-7 to 3e-6 in short runs.

### Code execution sandbox

Model-written code runs only in a Modal Sandbox with these properties:

- `block_network=True`, CPU only, and no volumes, secrets or HF token.
- `timeout` set to the smaller of 24 hours and the job's remaining launch budget, per
  [AGENTS.md](../../AGENTS.md).
- The trainer creates one sandbox per training call and records its ID in the run's
  deadline record. It terminates the sandbox in a `finally` block, and the deadline
  supervisor terminates it by ID if the trainer dies.

Each step sends the batch of programs to the sandbox, which runs them in parallel worker
processes:

- **Per-program limits:** about 5 s of CPU time, 512 MB of memory, a 64 KB output cap,
  and a fresh temporary working directory.
- **How each program runs:** the source is fed to `python -` on stdin, so `__file__` is
  undefined. The source is the solution followed by the tests. After the last test it
  prints a per-step random nonce, which the model never sees.
- **What counts as a pass:** exit code 0 and the nonce on stdout. A program that calls
  `sys.exit(0)`, skips the tests or crashes cannot pass.

Execution overlaps with the next step's math rollouts, so the GPU doesn't wait for the
CPU.

### Rough cost

These are estimates, not measured:

- **Sampling:** 512 sequences of at most 384 tokens with a KV cache take a few seconds
  on an H100 for a 124M model.
- **Gradient step:** about 300K tokens, around a second.
- **Code execution:** 256 programs on a 16-core sandbox take tens of seconds.

That comes to about 20–40 s per step, so 1,000 steps take 6–11 hours. That is one
24-hour-bounded call with checkpoints every 100 steps, roughly $30–50 of H100 time.

## Measurement

**Logged every step:**

- Reward per task.
- The distribution of per-prompt pass rates and the share of identical-reward groups.
- Mean answer length and truncation rate.
- Token entropy.
- KL to the starting checkpoint, replay loss and gradient norm.

**Every 100 steps:**

- Greedy accuracy on the 498 GSM8K validation problems from the SFT data.
- Pass@1 on MBPP validation (90 problems).
- Replay validation loss per source.

**At the end, with `analysis/evaluation`:**

- GSM8K and SVAMP.
- New `humaneval` and `mbpp` benchmark modules: EvalPlus test sets, greedy pass@1 and
  pass@10 at temperature 0.8, run in the same sandbox runner.
- The general benchmarks and held-out bits per byte.

Watch for these failure signs:

- **Reward hacking:** answers that pass the training checker but fail the stricter
  EvalPlus tests, or math answers without working.
- **Entropy collapse:** entropy near zero while reward plateaus.
- **Length blow-up:** truncation rate rising.

## Runs

| Run | Purpose | Success looks like |
| --- | --- | --- |
| S0 | Joint math + code SFT with replay | Code pass@1 above 0, math and general scores at r02 level, pool gate met |
| R1 | Math-only RL from S0, 300 steps | GSM8K rises; validates the loop and logging cheaply |
| R2 | Code-only RL from S0, 300 steps | MBPP pass@1 rises; validates the sandbox |
| R3 | Joint RL from S0, 1,000 steps | Both rise; general scores within the targets |
| Ablations on R1 | G = 8 vs 16; replay on vs off; LR 5e-7 / 1e-6 / 3e-6 | Pick settings for R3 |

## Planned code layout

| File | Role |
| --- | --- |
| `data/code_sft.py` | KodCode/MBPP `SFTDataset` with retained tests and decontamination |
| `data/rl_tasks.py` | RL prompt pools: prompt text, reference answer or tests, and the checker |
| `trainer/rollout.py` | Batched KV-cache sampling on the GPU, returning tokens and log-probabilities |
| `trainer/rewards.py` | Math-Verify reward; the client for the code sandbox |
| `trainer/rl.py` | The loop: prompt scheduling, advantages, loss, replay, checkpoints, metrics |
| `modal_scripts/modal_rl_train.py` | Launcher: 24-hour deadline, sandbox lifecycle, supervisor |
| `analysis/evaluation/benchmarks/{humaneval,mbpp}.py` | Code benchmarks with the sandbox runner |
| `rl/recipes/*.json` | Run recipes |

Checkpoints follow the SFT checkpoint layout, with a new format ID, so the
evaluation loader and playground can read them.
