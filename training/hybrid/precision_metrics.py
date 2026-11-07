"""Estimated model FLOPs against explicit dense H100 SXM precision ceilings.

These estimates exclude recurrence, elementwise, optimizer and conversion FLOPs.
They are not hardware-counter measurements of tensor-core activity.
"""

H100_BF16_FLOPS = 989e12
H100_FP8_FLOPS = 1978e12


def utilization(flops, fp8_flops, tokens_per_second, world):
    if not 0 <= fp8_flops <= flops:
        raise ValueError("FP8 FLOPs must be a subset of total model FLOPs")
    rate = tokens_per_second / world
    bf16_time = (flops - fp8_flops) / H100_BF16_FLOPS
    fp8_time = fp8_flops / H100_FP8_FLOPS
    return {
        "mfu": rate * flops / H100_BF16_FLOPS,
        "mfu_fp8_peak": rate * flops / H100_FP8_FLOPS,
        "mfu_mixed_precision": rate * (bf16_time + fp8_time),
        "estimated_model_tflops_per_gpu": rate * flops / 1e12,
        "estimated_fp8_flop_fraction": fp8_flops / flops,
    }


def fp8_flops_per_token(lm, config, loops=None):
    names = getattr(lm, "_fp8_linear_names", ())
    uses = config.block_uses(config.n_loops if loops is None else loops)
    weights = 0
    for name in names:
        count = 1
        if name.startswith("blocks."):
            count = uses[int(name.split(".")[1])]
        weights += count * lm.get_submodule(name).weight.numel()
    # Each matrix has forward, input-gradient and weight-gradient GEMMs.
    factor = 4 if getattr(lm, "_fp8_recipe", None) == "rowwise_with_gw_hp" else 6
    return factor * weights
