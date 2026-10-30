"""StableReinforce: corrections for answers our sampler wrote, which are slightly off-policy.

The sampler (cached one-token decoding, `trainer/rollout.py`) and the learner (a full
forward pass) compute probabilities with different kernels. So even the answers the
current weights just sampled come from a distribution `q` a little different from the
learner's `p`. Tinker's StableReinforce
(`../../../references/stable-reinforce-2026-10-25.md`) is REINFORCE with corrections for
that, which this hook adds to any algorithm training on our sampler's answers
(trajectory sources `sampler` and `reuse`):

- Score Centering (Marek and Ryabinin, arXiv:2609.20807), with `top_k` above 0: each
  token's score, the term the algorithm's loss pushes up (`Tokens.score`), becomes

      ∇ log p(y_t) − Σ_v q_v ∇ log p_v

  With respect to the logits that gradient is `onehot(y_t) − q`, against
  `onehot(y_t) − p` for REINFORCE: the dense `−p` part, which carries any bias between
  the two engines, cancels. When they agree exactly, both are equal. `q` is estimated
  from the sampler's `top_k` likeliest tokens, with the remaining mass spread over the
  other tokens like the learner's (detached) probabilities, the paper's top-k estimator;
  k = 32 matched exact centering in all its settings. Centering needs answers sampled
  at temperature 1.
- `sampler_kl_coef` adds the k3 estimate of KL(sampler ‖ learner) on the sampled tokens.

The hook logs `sampler_mismatch`, the mean sampler minus learner log-probability of the
sampled tokens. Settings: `"stable_reinforce": {"top_k": 32, "sampler_kl_coef": 0.0}`;
without that section the hook is off.
"""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class StableReinforceSettings:
    top_k: int = 32
    sampler_kl_coef: float = 0.0

    def __post_init__(self):
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not math.isfinite(self.sampler_kl_coef) or self.sampler_kl_coef < 0:
            raise ValueError("Invalid sampler KL coefficient")

    def check(self, settings):
        """Refuse RL `settings` whose answers the hook cannot correct."""
        if settings.trajectories.source == "file":
            raise ValueError("StableReinforce needs answers our sampler wrote")
        if self.top_k and settings.rollouts.temperature != 1:
            raise ValueError("Score centering needs sampling at temperature 1")


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


class StableReinforce:
    def __init__(self, settings):
        self.top_k, self.kl_coef = settings.top_k, settings.sampler_kl_coef
        self.mismatch, self.recorded = 0.0, 0

    @classmethod
    def attach(cls, settings):
        """A hook for RL `settings`, or None when StableReinforce is off."""
        if settings.stable_reinforce is None:
            return None
        return cls(settings.stable_reinforce)

    def apply(self, tokens, batch, chosen):
        """Correct micro-batch `tokens`, answers `chosen` of `batch`, with the sampler's
        records (`Batch.sampler`): center its score; returns the sampler KL loss term."""
        shape, device = tokens.targets.shape, tokens.targets.device
        sampled = torch.zeros(shape, device=device)
        recorded = torch.zeros(shape, dtype=torch.bool, device=device)
        top_ids = torch.zeros((*shape, self.top_k), dtype=torch.long, device=device)
        top_logprobs = torch.zeros((*shape, self.top_k), device=device)
        for b, i in enumerate(chosen):
            if batch.sampler is None or batch.sampler[i] is None:
                raise RuntimeError("StableReinforce needs the sampler's records of every answer")
            # Answer tokens are targets from the prompt's last position on.
            at = len(batch.rows[i]["prompt_ids"]) - 1
            logprobs, ids, values = batch.sampler[i]
            sampled[b, at : at + len(logprobs)] = logprobs
            recorded[b, at : at + len(logprobs)] = True
            if self.top_k:
                top_ids[b, at : at + len(ids)] = ids
                top_logprobs[b, at : at + len(ids)] = values
        mask = tokens.mask & recorded
        if self.top_k:
            tokens.score = centered(tokens.logits, tokens.targets, top_ids, top_logprobs)
        self.mismatch += ((sampled - tokens.logp.detach()) * mask).sum().item()
        self.recorded += int(mask.sum())
        if not self.kl_coef:
            return 0.0
        drift = sampler_kl(tokens.logp, sampled, mask)
        return self.kl_coef * (tokens.scale[:, None] * drift).sum()

    def statistics(self):
        """{"sampler_mismatch": mean over the tokens since the last call}."""
        values = {"sampler_mismatch": self.mismatch / self.recorded} if self.recorded else {}
        self.mismatch, self.recorded = 0.0, 0
        return values
