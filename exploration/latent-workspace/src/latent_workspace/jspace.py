"""Exact batched Jacobian estimator; sparse vocabulary readout conditioning.

Estimator follows anthropics/jacobian-lens (Apache-2.0), fitting.py, reference
581d398613e5602a5af361e1c34d3a92ea82ba8e. This independent implementation consumes
training token IDs directly and uses one-based post-block residual indices.
"""

import time
from pathlib import Path

import torch
from torch.nn import functional as F


def fit_jacobians(backbone, prefixes, *, deadline, checkpoint_path=None, on_checkpoint=None):
    cfg = backbone.cfg
    model = backbone.model.model
    hidden = backbone.hidden_size
    sources = [i for i in (cfg.selected_layers or cfg.layers[:-1]) if i != cfg.layers[-1]]
    sums = {i: torch.zeros(hidden, hidden) for i in sources}
    counts = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        saved = torch.load(checkpoint_path, weights_only=True)
        if saved["layers"] != list(cfg.layers):
            raise ValueError("J-lens calibration layers mismatch")
        sums, counts = saved["sums"], saved["counts"]
    for ids in prefixes[counts : cfg.jspace_prompts]:
        ids = ids[ids >= 0]
        if time.monotonic() > deadline:
            raise TimeoutError("time budget exhausted fitting J-lens")
        captures, hooks = {}, []

        def capture(index, captures=captures):
            def hook(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                if index == min(sources):
                    value = value.detach().requires_grad_(True)
                    output = (value, *output[1:]) if isinstance(output, tuple) else value
                captures[index] = value
                return output

            return hook  # noqa: B023 - created and registered within this iteration

        try:
            for i in cfg.layers:
                hooks.append(model.layers[i - 1].register_forward_hook(capture(i)))
            batch = cfg.jspace_dim_batch
            with torch.enable_grad():
                model(
                    ids[None].expand(batch, -1).to(next(model.parameters()).device), use_cache=False
                )
                final = captures[cfg.layers[-1]]
                # Reference excludes the first 16 and last position.
                valid = slice(16, ids.numel() - 1)
                if ids.numel() < 18:
                    raise ValueError("J-lens requires at least 18 prefix tokens")
                for start in range(0, hidden, batch):
                    if time.monotonic() > deadline:
                        raise TimeoutError("time budget exhausted fitting J-lens")
                    n = min(batch, hidden - start)
                    cotangent = torch.zeros_like(final)
                    for row in range(n):
                        cotangent[row, valid, start + row] = 1
                    grads = torch.autograd.grad(
                        final,
                        [captures[i] for i in sources],
                        grad_outputs=cotangent,
                        retain_graph=start + batch < hidden,
                    )
                    for i, grad in zip(sources, grads):
                        sums[i][start : start + n] += grad[:n, valid].float().mean(1).cpu()
            counts += 1
            if checkpoint_path:
                path = Path(checkpoint_path)
                tmp = path.with_suffix(".tmp")
                torch.save({"sums": sums, "counts": counts, "layers": list(cfg.layers)}, tmp)
                tmp.replace(path)
                if on_checkpoint:
                    on_checkpoint()
            print(
                f"J-lens calibration {counts}/{min(len(prefixes), cfg.jspace_prompts)}", flush=True
            )
        finally:
            for hook in hooks:
                hook.remove()
    matrices = [sums[i] / counts if i in sums else torch.eye(hidden) for i in cfg.layers]
    return torch.stack(matrices)


@torch.no_grad()
def readout(backbone, residual):
    """Top-k J-lens concepts -> weighted frozen token vectors, per layer/position.

    This is an explicit compressed J-lens readout, NOT a claim to recover the
    paper's complete sparse nonnegative residual-space frame.
    """
    transport = backbone.jacobians.to(residual.device)
    mapped = torch.einsum("ldh,blth->bltd", transport, residual.float())
    weight = backbone.model.get_output_embeddings().weight
    outputs = []
    # Bound vocabulary-logit peak memory to one depth and token position.
    active = set(backbone.cfg.selected_layers or backbone.cfg.layers)
    for depth in range(mapped.shape[1]):
        if backbone.cfg.layers[depth] not in active:
            outputs.append(torch.zeros_like(residual[:, depth]))
            continue
        positions = []
        for position in range(mapped.shape[2]):
            normalized = backbone.model.model.norm(mapped[:, depth, position].to(weight.dtype))
            logits = F.linear(normalized, weight).float()
            scores, indices = logits.topk(backbone.cfg.jspace_top_k, dim=-1)
            vectors = F.embedding(indices, weight).float()
            positions.append((scores.softmax(-1)[..., None] * vectors).sum(-2))
        outputs.append(torch.stack(positions, dim=1))
    return torch.stack(outputs, dim=1).to(residual.dtype)
