# StableReinforce is Reinforce + Score Centering

Reviewed 2026-10-25. Martin Marek (Recursive), [thread on X](https://x.com/mrtnm/status/2114145574987702525)
(2026-10-25), reverse-engineering Tinker's
[StableReinforce](https://tinker-docs.thinkingmachines.ai/tinker/losses/stable-reinforce/)
loss (alpha). The method it identifies is
[Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807)
(Marek and Ryabinin, arXiv:2609.20807). The results below are the authors'; we have
not reproduced them.

## What Tinker documents

StableReinforce is "a REINFORCE variant that stays stable even when the sampling
policy differs from the policy being trained". Sampling must run with
`record_stability_info=True`, and training data references sampled sequences by
`sequence_id`. The loss adds a sampler–learner KL term (`kl_coeff`, default 0.01;
`kl_estimator` `k1` or `k3`), not a KL to a reference model. Tinker reports
"extremely stable training, even under highly off-policy async RL and low precision
numerics". The docs do not say what the stability info is or how the loss uses it.

## How the loss was recovered

1. **Raw gradients from an Adam step.** Tinker has no SGD, so the step uses Adam
   with ε = 10,000. That ε dominates the second-moment term, so
   `Δw ≈ −lr · g / ε`, and downloading the updated weights recovers `g`.
2. **Exact logit gradients from a rank-1 LoRA.** With a rank-1 adapter `B · A` on
   the unembedding of Qwen3.5-4B, the gradient with respect to `B` equals the
   gradient with respect to the logits up to a scale.
3. **Only 8 non-zero logit gradients.** For a single-token update, 8 logits get a
   gradient. Plain Reinforce puts a non-zero gradient on every logit, because each
   token's score contains the dense softmax part `−p`.
4. **Hypothesis.** Score Centering subtracts the sampler's expected score from the
   sampled token's score. If that expectation is estimated from 8 tokens drawn from
   the sampler, the shared `−p` part cancels exactly and only those tokens keep a
   gradient.
5. **Verification.** Reinforce + Score Centering with an 8-sample estimate,
   implemented as a Tinker custom loss, gives gradients that match StableReinforce
   to numerical precision across all layers.

So the "stability info" is presumably the sampler-side information needed for the
expected-score term, such as the 8 sampled correction tokens. The thread does not
cover the KL term.

## Score Centering

For sampled token `y_t` with score `s_v = ∇θ log p_v`, the centered score is
`s_{y_t} − Σ_v q_v s_v`, where `q` is the sampler (inference-engine) distribution.
As a loss, with stop-gradient on `q`:

    L = −R · (log p_{y_t} − Σ_v sg[q_v] · log p_v)

The correction is exactly zero when trainer and sampler agree. The paper attributes
RL instability under training–inference mismatch to *drift*, a persistent bias
between the engines that accumulates every step, and derives the correction to cancel it.

Estimating `Σ_v q_v s_v`:

| Estimator | Used by | Properties |
| --- | --- | --- |
| Exact sum over the vocabulary | Paper (reference) | Needs the full sampler distribution, about 608 KB per token in fp32 at a 152K vocabulary. |
| Sampler top-k, tail modeled by the rescaled trainer distribution | Paper | Deterministic, slightly biased. k = 128 default; k = 32 matched full score centering in all tested settings, at under 1% wall-clock overhead. |
| Mean score over 8 tokens drawn from the sampler | Tinker, per the thread | Unbiased, higher variance. |

Paper results (Qwen3 0.6B on Countdown, Qwen3-30B-A3B-Base on INTELLECT-2 math):

- Under the largest synthetic weight noise at 0.6B, only score centering (alone or
  with TIS/MIS importance sampling) trained stably.
- With an FP4 KV cache at 30B, plain policy gradient collapsed within 200 steps;
  score centering reached about 52% training accuracy, TIS about 51%.
- With an INT8 sampler and INT4 KV cache at 30B, score centering reached about 30%,
  TIS about 12%, and the other methods ended below 5%.
- With staleness (sampler updated every 64 steps), score centering composed with
  TIS or MIS did best.

## What this means for our models

The [training plan](../README.md) schedules verifiable-reward RL after SFT, with
GRPO- or PPO-style methods. If those rollouts come from a separate or quantized
inference engine (for example our own phone kernels), score centering is a cheap
additive correction to compare against importance sampling. The top-k form needs
only the sampler's top-k log-probs per token, which samplers such as vLLM can return.
Nothing here is implemented or measured in this repository.

The ε trick is also a general probe: with a large Adam ε, a black-box fine-tuning
API returns `−lr · g / ε`, which exposes the raw gradient.
