"""Simulated weight quantisation: how much accuracy low-bit weights would cost.

`round_weights_` replaces every `nn.Linear` weight under a module, in place, by its
symmetric round-to-nearest quantisation to `bits`-bit integers with one scale per group
of `group` consecutive input weights (a row's last group may be shorter), dequantised back to the weight's dtype. This is
the arithmetic of block-wise integer weight formats (such as 4-bit blocks of 32), so
evaluating with it measures their accuracy; it does not make inference smaller or faster.
"""

import torch


def round_weights_(module, bits, group=32):
    """Quantise and dequantise every Linear weight under `module`; returns the number of
    layers and weights changed."""
    if bits < 2:
        raise ValueError(f"bits must be at least 2, not {bits}")
    top = 2 ** (bits - 1) - 1
    layers = weights = 0
    for layer in module.modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        weight = layer.weight.data
        rows, columns = weight.shape
        padded = -columns % group  # zeros change no group's largest magnitude
        blocks = torch.nn.functional.pad(weight.float(), (0, padded)).view(rows, -1, group)
        scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-12) / top
        rounded = (blocks / scale).round().clamp(-top - 1, top) * scale
        layer.weight.data = rounded.view(rows, -1)[:, :columns].to(weight.dtype)
        layers += 1
        weights += weight.numel()
    if not layers:
        raise ValueError("No Linear layers to quantise")
    return layers, weights
