"""Per-token scores the policy gradient pushes up, and the sampler–learner KL.

Plain REINFORCE pushes up the score `∇ log p(y_t)` of each sampled token, weighted by
its advantage. Score Centering (Marek and Ryabinin, arXiv:2609.20807; the method behind
Tinker's StableReinforce, see `../../references/stable-reinforce-2026-10-25.md`)
subtracts the sampler's expected score, so the token's term becomes

    ∇ log p(y_t) − Σ_v q_v ∇ log p_v

where `q` is the distribution the token was actually sampled from. With respect to the
logits this gradient is `onehot(y_t) − q`, against `onehot(y_t) − p` for REINFORCE: the
dense `−p` part, which carries any bias between the sampler's numerics and the
learner's, cancels. When sampler and learner agree exactly, both are equal.

`q` is estimated from the sampler's `k` likeliest tokens, with the remaining mass spread
over the other tokens in proportion to the learner's own (detached) probabilities, the
paper's top-k estimator; k = 32 matched exact score centering in all its settings.
"""

from torch.nn import functional as F


def centered(logits, targets, top_ids, top_logprobs):
    """Centered score per target `[batch, time]`, differentiable in `logits`
    `[batch, time, vocab]`; `top_ids`/`top_logprobs` `[batch, time, k]` are the sampler's
    top-k at each target position. Only its gradient is meaningful."""
    logp = F.log_softmax(logits.float(), -1)
    picked = logp.gather(-1, targets.clamp(min=0)[..., None])[..., 0]
    probabilities = logp.detach().exp()
    top = logp.gather(-1, top_ids)
    learner_top = probabilities.gather(-1, top_ids)
    sampler_top = top_logprobs.exp()
    # Learner-weighted log-probabilities of every token, then of the non-top-k ones.
    tail = (probabilities * logp).sum(-1) - (learner_top * top).sum(-1)
    tail_weight = (1 - sampler_top.sum(-1)).clamp(min=0) / (1 - learner_top.sum(-1)).clamp(min=1e-6)
    return picked - ((sampler_top * top).sum(-1) + tail_weight * tail)


def sampler_kl(learner_logprobs, sampler_logprobs, mask):
    """k3 estimate of KL(sampler ‖ learner) per masked sampled token, differentiable in
    the learner's log-probabilities: `r − 1 − log r` with `r = p / q`."""
    delta = learner_logprobs - sampler_logprobs
    return (delta.exp() - 1 - delta) * mask
