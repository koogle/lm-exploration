"""CIG-SFT (arXiv:2610.12995v1), equations 7 and 19–20.

Reference scores are frozen; current-policy token weights must be detached.
Normalization is by supervised token count, never by the sum of weights.
"""

import math

import torch
from torch.nn import functional as F


def information_ratio(conditional_nll, task_free_nll):
    return ((task_free_nll - conditional_nll) / (task_free_nll + 1e-8)).clamp(0, 1).detach()


def weights(nll, information, proximity=0.3):
    if not math.isfinite(proximity) or proximity < 0.01:
        raise ValueError("CIG proximity must be finite and >= 0.01")
    # w = a/(1+a), a = p * expm1(c/lambda). Log-space avoids overflow.
    c = information.detach().clamp(0, 1) / proximity
    positive = c > 0
    safe = c.clamp_min(torch.finfo(c.dtype).tiny)
    log_expm1 = safe + torch.log(-torch.expm1(-safe))
    return torch.where(positive, torch.sigmoid(-nll.detach() + log_expm1), 0).detach()


def weighted_loss(logits, targets, information, proximity=0.3):
    nll = F.cross_entropy(
        logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="none"
    ).view_as(targets)
    mask = targets != -100
    weight = weights(nll, information, proximity)
    return (nll * weight * mask).sum(), mask.sum()


@torch.inference_mode()
def reference_cache(model, splits, raw_rows, tokenizer, context):
    """Score identical gold answer tokens with/without task; never train the reference.

    Empty task preserves language and prompt scaffolding, removes instruction and
    broken code. Cache is recomputed from the original parent on resume.
    """
    from training_data.repair_sft import encode

    model.eval()
    by_id = {r["id"]: r for r in raw_rows}
    cache = {}
    for row in splits["train"]:
        blank = dict(by_id[row["id"]], instruction="", broken="")
        task_free = encode(blank, tokenizer, context)
        scores = []
        for encoded in (row, task_free):
            x = torch.tensor([encoded["x"]], device="cuda")
            y = torch.tensor(encoded["y"], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)[0].float()
            nll = F.cross_entropy(logits, y, ignore_index=-100, reduction="none")
            scores.append(nll[y != -100])
        full_targets = [t for t in row["y"] if t != -100]
        if full_targets != [t for t in task_free["y"] if t != -100]:
            raise ValueError("Task-free scoring changed answer-token alignment")
        c = torch.zeros(len(row["y"]), device="cuda")
        c[torch.tensor(row["y"], device="cuda") != -100] = information_ratio(*scores)
        cache[row["id"]] = c.cpu()
    model.train()
    return cache
