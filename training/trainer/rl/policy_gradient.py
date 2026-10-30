"""Pieces the policy-gradient algorithms share: group-relative advantages and the
(optionally ratio-weighted and clipped) policy-gradient loss with a plain or centered
score (`scores.py`)."""

import torch

from trainer.rl import scores

SCORES = ("plain", "centered")


def check_score(algorithm):
    if algorithm.score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}")
    if type(algorithm.centering_top_k) is not int or algorithm.centering_top_k < 1:
        raise ValueError("centering_top_k must be a positive integer")


def group_advantages(rewards, group, scale_by_std=False):
    """Reward minus the mean reward of its group of `group` consecutive answers, divided
    by the group's standard deviation with `scale_by_std`."""
    result = []
    for start in range(0, len(rewards), group):
        chunk = rewards[start : start + group]
        mean = sum(chunk) / len(chunk)
        spread = (sum((r - mean) ** 2 for r in chunk) / (len(chunk) - 1)) ** 0.5
        divisor = spread + 1e-6 if scale_by_std else 1.0
        result += [(r - mean) / divisor for r in chunk]
    return result


def loss(tokens, clip=None, score="plain"):
    """−Σ scale · r · A · score over answer tokens, with r = p / p_old (detached), so its
    gradient is −r·A·∇log p for the plain score. The centered score is Score Centering's
    (`scores.centered`, from the sampler's top tokens in `tokens`). With `clip`, PPO's
    clipped surrogate: tokens whose ratio left 1 ± clip in the advantage's direction get
    no gradient. Returns (loss, statistics)."""
    ratio = (tokens.logp - tokens.old_logp).detach().exp()
    advantage = tokens.advantage[:, None]
    clipped = torch.zeros_like(tokens.mask)
    if clip is not None:
        clipped = ((advantage > 0) & (ratio > 1 + clip)) | ((advantage < 0) & (ratio < 1 - clip))
    weight = tokens.scale[:, None] * ratio * advantage * (~clipped & tokens.mask)
    value = tokens.logp
    if score == "centered":
        value = scores.centered(
            tokens.logits, tokens.targets, tokens.sampler_top_ids, tokens.sampler_top_logprobs
        )
    return -(weight * value).sum(), {"clipped_tokens": (clipped & tokens.mask).sum().item()}
