# Looped hybrid LM: implementation plan

A looped (latent-recurrent) variant of the [720M hybrid](HYBRID_720M.md): the middle
blocks share weights and run `k` times per token, with `k` chosen per request. It
follows Elastic Latent Recurrence (ELR, [arXiv:2610.12582](https://arxiv.org/abs/2610.12582)),
whose Gated DeltaProduct hybrid is the closest published looped model to our KDA
hybrid. The evidence, Dory1.0-4B and the comparison with our runs are in
[references/looped-lms-2026-11-03.md](../references/looped-lms-2026-11-03.md).

The question is ELR's, asked at our scale: can one checkpoint match the plain 720M
hybrid at one loop and gain on reasoning and code at two to four loops, for about
1.2× the training compute?

## Stages

| Stage | Scope | State |
| --- | --- | --- |
| 1 | Model: prelude / recurrent / coda split, input re-injection, adaptive-delta update, `loops` per forward call, FLOPs accounting | Implemented, CPU-tested |
| 2 | Trainer: mixed-depth sampling with one-loop rank overrides, delayed switch, per-`k` held-out loss; converting a plain export (`loop_retrofit`) | Implemented, CPU-tested |
| 2b | Conversion test: Hybrid v2 base (mid32k export), 1B tokens, plain vs looped | Launched 2026-11-03 |
| 3 | Matched pilot: baseline vs looped 720M at 10B tokens | Proposed; needs approval and budget |
| 4 | Decoding: per-loop and last-loop caches in `generate.py`; benchmarks at `k = 1..4` | Proposed |
| 5 | Loop-conditioned modulation grafted at the switch | Proposed, after stage 3 succeeds |

## Stage 1: the model (implemented)

`HybridConfig` takes `loop_prelude`, `loop_coda` and `n_loops`; blocks
`[loop_prelude, n_layer - loop_coda)` are the shared recurrent group, and
`n_loops = 1` (the default) is the existing model. `HybridLM.trunk`:

```
h = prelude(embed(tokens))
state = h
repeat k times:                       # k = loops argument, default n_loops
    entry = state + h                 # re-inject the prelude output
    proposal = recurrent_blocks(entry)
    beta = sigmoid(W entry + b) * softplus(gamma)
    state = state + beta * (proposal - state)
output = coda(state)
```

- `LoopUpdate` holds W (`d × d`, Muon), b and gamma (AdamW, no decay), zero-initialized
  so every step starts at ≈ 0.35. For the 720M width that is 2.36M parameters (0.3%).
- `forward(tokens, labels, mtp_labels, loops=k)` runs any `k ≥ 1` with the same
  weights; the MTP module reads the coda output as before.
- Document masking, RoPE and KDA state resets are unchanged; every loop re-runs KDA
  from a zero state over the whole sequence, as in training.
- Residual-output initialization stays `1/sqrt(2 · n_layer)`: at one loop each block
  runs once, as in the baseline.
- `flops_per_token` counts recurrent blocks and the loop gate once per loop at
  `n_loops`.
- `generate.py` refuses looped models until stage 4.
- Tests: [`tests/test_hybrid_looped.py`](../tests/test_hybrid_looped.py) checks the
  forward pass against a hand-written loop at `k = 1, 2, 3, 5`, weight sharing,
  causality and document isolation, the initial step size, gradients and optimizer
  roles, and the FLOPs count.

Proposed 720M layout: 4 / 8 / 4 blocks (ELR puts about the middle half in the
recurrent block). The recurrent blocks 4–11 hold 6 KDA and 2 full-attention layers
(blocks 7 and 11). Non-embedding parameters are 469M; the recurrent group is 235M,
so `k = 2` touches about 1.5× and `k = 4` about 2.5× the one-loop weights.

## Stage 2: the trainer (implemented)

Run-config fields `one_loop_ranks` and `loop_start_tokens` (see `RunConfig`); the
loop count of each microbatch comes from `HybridFamily.loop_count`. Held-out
evaluation reports `heldout` at `n_loops` and `heldout_by_loops` at every smaller
count, plus the loop update's base step; metrics log the mean loop count and MFU at
that count. `init_from` loads a plain export into a looped model (only `loop_update`
is new). With `loop_retrofit` the state starts at zero and the step at 1, so the
converted model at one loop computes exactly its parent, and two loops start as
"rerun the recurrent blocks on their own output plus the prelude output".

Design notes:

- **Loop count per microbatch.** `k ~ Uniform{1..K}` with `K = 4`, drawn from a seeded
  generator and shared by all ranks (synchronized), with a rotating set of ranks
  overridden to `k = 1`. On 8 ranks one override (`f = 1/8`) gives
  `p_1 = 1/K + (1 − 1/K) f = 0.34` and `k̄ = f + (1 − f)(K + 1)/2 = 2.31`.
- **Delayed switch.** `loop_start_fraction` in the run config: `k = 1` before it,
  mixed depth after. ELR's 75% start costs ~1.2× baseline compute and keeps 84% of the
  deep-loop gain on GPT-0.5B. The optimizer state and learning-rate schedule carry over.
- **Interfaces.** The family's training step passes `loops` to `model(...)`;
  `flops_per_token` and MFU use the realized `k`; logs record `k` per step.
- **Compilation.** The loop is Python, so `torch.compile` traces one graph per `k`
  (four graphs); compiled blocks would avoid that if recompilation time matters.
  DDP sees every parameter used at any `k`.
- **Evaluation.** Held-out loss per source at `k = 1..K` at every evaluation step,
  plus the step size `beta` (mean and spread) per loop.

## Conversion test: can the Hybrid v2 base loop?

[`workflows/hybrid-720m-loop-test.json`](../workflows/hybrid-720m-loop-test.json)
continues the [mid32k export](HYBRID_V2.md#mid-training-run-hybrid-mid32k-v1) for 1B
tokens of the `hybrid-mid32k-v1` recipe read in 8K windows (run-config `context`; 1M
tokens per step, seed 4321, mid32k's learning rates with linear decay), as two
branches from the same parent on 8 × H100 each. Workflow training stages run one
after another, so each branch is its own launch (`--start` / `--end` with the
branch's name) under run IDs `hybrid-720m-loop-test-02a` (control) and `-02b`
(looped):

| Branch | Model |
| --- | --- |
| `control` | plain continuation |
| `looped` | 4 / 8 / 4 blocks, `loop_retrofit`, `n_loops = 4`, one of eight ranks at `k = 1` (`p_1 = 0.34`, `k̄ = 2.31`) |

The looped branch costs 1.45× the control's FLOPs per token (`flops_per_token` at
`k̄`). Estimated at ~450K tokens/s for the control (mid32k measured 513K at 32K):
about 40 minutes of training for the control and 55 for the looped branch, plus
start-up, compilation and evaluation, so about 1.5 hours of wall time and about 20
H100-hours in total. Each runs in one 4-hour session under the workflow's deadline
supervisor.

Launch `hybrid-720m-loop-test-01` stopped at the control's first training step:
the recipe's 32K windows exceeded the model's 8K `block_size` (the step-0 held-out
evaluation, which uses the model's context, had run). The looped branch never started.

It passes if the looped model at `k = 1` stays within 0.01 nats of the control on
every held-out source and `k = 2..4` are below the control, most on code, math and
reasoning traces. 1B tokens is far shorter than ELR's 250B-token mixed-depth phase,
so a null result here does not rule out looping at a larger budget.

## Stage 3: matched pilot

Two runs from scratch on the `hybrid-40b-v1` recipe with identical data order, seed
and optimizer settings:

| Run | Model | Tokens | Estimated 8 × H100 hours |
| --- | --- | ---: | ---: |
| Baseline | 720M hybrid, `n_loops = 1` | 10B | ~7.5 (at the measured 372.7K tokens/s) |
| Looped | 720M, 4 / 8 / 4, `K = 4`, switch at 75% | 10B | ~9 |

The 7.5 h is measured throughput; the 9 h assumes the looped run's throughput scales
with `k̄` and is not yet measured. Each job runs under the repository's 24-hour Modal
cap with a remote deadline supervisor; nothing launches without approval.

Gates:

1. One-loop held-out loss within 0.01 nats/token of the baseline on every source.
2. Loss falls monotonically from `k = 1` to `k = 4`; report the `k = 4` gain.
3. Benchmarks (the existing lm-evaluation-harness suite) at `k = 1, 2, 4`: GSM8K,
   HumanEval and MBPP are where ELR's gains concentrate; MMLU and knowledge tasks are
   not expected to move.

A failed gate 1 stops the plan for a decision; the run and its numbers go to the HF
results archive either way.

## Stage 4: decoding

`Decoder` and `_prefill_row` iterate the unrolled schedule (prelude, `k` copies of the
recurrent blocks, coda) instead of `lm.blocks`, with the loop update between copies:

- **Per-loop caches.** Each recurrent attention layer gets `k` KV caches and each
  recurrent KDA layer `k` states and convolution histories. KDA state is fixed-size, so
  this costs `k ×` a few MiB; only the two recurrent full-attention layers grow with
  context.
- **Last-loop caches.** One KV cache per recurrent attention layer, written by every
  loop, so earlier tokens are read at their final loop. Koi-4B lost 0–3 points this way
  at `k = 4` while keeping one-loop memory. KDA keeps per-loop state.
- One CUDA graph per `(rows, capacity, k)`.
- Measured on the pilot: decode latency, output length and repeated 4-grams at each
  `k` with the [sampling toolkit](../analysis/sampling/README.md).

The `k = 1` model can then draft for `k = 4` (self-speculative decoding with shared
weights), measured as acceptance rate and speed-up.

## Stage 5: loop conditioning

ELR's progress/budget modulation (AdaLN-style scale, shift and output gate plus a
gated low-rank adapter per recurrent and coda layer), added at the switch with
zero-initialized heads so the model starts as its one-loop checkpoint. On GPT-0.5B it
cost 6.3% more parameters and lowered `k = 5` loss by 0.022. Only after stage 3 shows
gains without it.
