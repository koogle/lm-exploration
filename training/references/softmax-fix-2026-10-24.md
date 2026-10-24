# A fix for softmax and cross-entropy

Reviewed 2026-10-24. Barron, Gnanapragasam and Tschannen (Google DeepMind),
[A Fix for Softmax and Cross-Entropy](https://jonbarron.info/softmaxfix/)
(arXiv:2610.31642). The results below are the authors'. They were measured on
grokking tasks with small Transformers and MLPs, not on language-model pretraining.

## The two bugs

**Softmax Collapse.** Standard log-softmax subtracts the max logit, so the
denominator is `1 + e^-m`, where `m` is the target logit minus the log-sum-exp of
the other logits. Past the precision cliff (m ≈ 5.5 in bf16, 16.6 in fp32, 36.7 in
fp64), the `e^-m` term is lost when it is added to 1. The loss and the target
gradient then round to 0, but the non-target gradients stay positive, so the
gradients no longer sum to zero. With Adam, the second moment decays while the
target gradient is zero. When an example drops back below the cliff, its gradient
is divided by a tiny `sqrt(v)`, and the step spikes the loss. This is the
"slingshot" seen in grokking.

**Softmax Cancellation.** For distillation and KL terms, `q - p` is formed by
subtracting two separately rounded softmaxes. When the student is close to the
teacher this loses precision, and the gradient is non-zero even at `z_S = z_T`.

## The fix

- Log-softmax and hard-target CE: put an exact 0 at the max (or target) slot, leave
  that slot out of the log-sum-exp to get `m`, and return `shifted - softplus(-m)`.
  The loss is `softplus(-m)`. Autodiff gives the target the negated sum of the other
  gradients, which stays accurate down to exponent underflow (m ≈ 88 in fp32).
- Soft-target CE: compute `q - p` from `δ = z_S - z_T` in log space with
  `expm1`/`log1p`, and swap that in as the backward pass.

## Reported results

- The fixed CE removes slingshot spikes. It groks faster with weight decay and does
  not collapse without it (61% → 23% test accuracy with standard CE versus 85% with
  the fix, on a² + b² mod 97).
- With AdamW, margins settle near `log(1/ε)`: 18.4 for the default ε = 1e-8, just
  above the fp32 cliff. So the default ε is what makes standard fp32 CE spike.
- fp32 and bf16 with the fixed CE match fp64 standard CE on four grokking benchmarks.
- When a student is initialized to its teacher, the fixed soft-target CE keeps KL
  at 0. Standard CE drifts in bf16.
- The authors note that such extreme margins are uncommon in typical large-scale
  training.

## What this means for our models

Our trainer runs `F.cross_entropy` under bf16 autocast, and autocast computes CE in
fp32, so the relevant cliff is about 16.6 nats of margin. A per-token loss of
`softplus(-16.6) ≈ 6e-8` happens only on near-certain tokens, such as boilerplate
code or memorized repeats. Whether our 124M or 10M models put a meaningful share of
tokens past the cliff has **not been measured**. That is the first check before
treating the cliff as a cause of any of our [negative results](../NEGATIVE_RESULTS.md).

[`trainer/stable_softmax.py`](../trainer/stable_softmax.py) ports the authors' JAX
code to PyTorch: `log_softmax`, `softmax`, `cross_entropy` (with `ignore_index`
and reductions) and `soft_target_cross_entropy`.
[`tests/test_stable_softmax.py`](../tests/test_stable_softmax.py) reproduces both of
the paper's worked examples, and checks fp32 against the exact gradient up to
m = 80 and agreement with the standard functions on ordinary logits. The module is
not wired into the training loop. In eager mode it materializes several N×V
temporaries. One rough CPU measurement (2048 × 32768 fp32, forward and backward)
took 0.86 s against 0.31 s for `F.cross_entropy`. Measure on GPU (eager and
`torch.compile`) before using it for pretraining.

[`trainer/kernels/stable_ce.cu`](../trainer/kernels/stable_ce.cu) is a fused CUDA
kernel for the hard-target loss, wrapped by
[`trainer/stable_ce_cuda.py`](../trainer/stable_ce_cuda.py). It runs one block per row.
Its forward pass streams the row once with an online log-sum-exp over `z_j - z_t`
(j ≠ t) and keeps one float per row. Its backward pass writes
`g_j = exp(z_j - z_t - loss)` and `g_t = -exp(lse - loss)` straight into the gradient,
so the only N×V allocation is the gradient itself. Half/bf16 logits are read directly
and computed in fp32. The extension is JIT-built with `torch.utils.cpp_extension` and
needs `nvcc`; it must be built without `--use_fast_math`. It compiles for sm_80/sm_90
against torch 2.14 / CUDA 13.0. It has **not yet run on a GPU**:
[`tests/test_stable_ce_cuda.py`](../tests/test_stable_ce_cuda.py) (skipped without
CUDA) and a throughput comparison with `F.cross_entropy` are still pending.

Proposed experiment, not run:
1. Log the fraction of training tokens with margin > 16.6 from a published checkpoint.
2. If that share is non-trivial, run a matched short continuation with the fixed CE.
