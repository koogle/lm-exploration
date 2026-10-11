"""Equal-capacity probes measure accessible future-token information by layer."""

import time

import torch
from torch.nn import functional as F

from .models import BlockHead, loss_and_logits


@torch.no_grad()
def cached_metrics(head, backbone, memory, targets, cfg, device):
    head.eval()
    sums = torch.zeros(cfg.block_size, device=device)
    hits = torch.zeros_like(sums)
    repeated = 0
    for start in range(0, len(targets), cfg.batch_size):
        target = targets[start : start + cfg.batch_size].to(device)
        _, logits, _ = loss_and_logits(
            head,
            backbone,
            memory[start : start + cfg.batch_size].to(device),
            target,
            all_masked=True,
        )
        sums += F.cross_entropy(logits.float().transpose(1, 2), target, reduction="none").sum(0)
        predictions = logits.argmax(-1)
        hits += (predictions == target).sum(0)
        repeated += int((predictions == predictions[:, :1]).all(-1).sum())
    return {
        "nll_by_position": (sums / len(targets)).tolist(),
        "accuracy_by_position": (hits / len(targets)).tolist(),
        "same_prediction_at_all_positions": repeated / len(targets),
    }


def layer_scan(
    backbone,
    train_memory,
    train_targets,
    valid_memory,
    valid_targets,
    cfg,
    device,
    deadline,
    *,
    domain_ranges=None,
    on_layer=None,
):
    results = {}
    conditions = [str(layer) for layer in cfg.layers]
    if cfg.scan_attention:
        conditions += [f"attention{layer}" for layer in cfg.layers]
    for condition in conditions:
        torch.manual_seed(cfg.seed)
        head = BlockHead(backbone.hidden_size, cfg, f"probe:{condition}").to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.learning_rate)
        sampler = torch.Generator().manual_seed(cfg.seed)
        for _ in range(cfg.scan_steps):
            if time.monotonic() > deadline:
                raise TimeoutError("time budget exhausted during layer scan")
            index = torch.randint(len(train_targets), (cfg.batch_size,), generator=sampler)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = loss_and_logits(
                head, backbone, train_memory[index].to(device), train_targets[index].to(device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        metric = cached_metrics(head, backbone, valid_memory, valid_targets, cfg, device)
        future = metric["nll_by_position"][1:] or metric["nll_by_position"]
        metric["future_nll"] = sum(future) / len(future)
        metric["parameters"] = sum(p.numel() for p in head.parameters())
        metric["steps"] = cfg.scan_steps
        if domain_ranges:
            metric["validation_by_domain"] = {
                domain: cached_metrics(
                    head, backbone, valid_memory[a:b], valid_targets[a:b], cfg, device
                )
                for domain, (a, b) in domain_ranges.items()
            }
        results[condition] = metric
        if on_layer:
            on_layer(results)
        print(f"Layer scan {condition}: future NLL {metric['future_nll']:.4f}", flush=True)
        del head, optimizer
    ranking = sorted(cfg.layers[:-1], key=lambda layer: results[str(layer)]["future_nll"])
    return {
        "layers": results,
        "attention_ranking": sorted(
            cfg.layers, key=lambda layer: results[f"attention{layer}"]["future_nll"]
        )
        if cfg.scan_attention
        else [],
        "intermediate_ranking": ranking,
        "selection_metric": "validation mean NLL over offsets 2..block_size",
        "interpretation": "Accessible information under this finite-budget probe, not intrinsic information.",
    }
