"""Benchmark-only Hopper FP8 linears; recurrence and vocabulary loss stay unchanged."""

import dataclasses

from torch import nn


def eligible(module, name, scope="all"):
    # Retain small gate/normalization-related maps in BF16. Cover the production
    # MLPs and large attention projections, including MTP's full-attention block.
    return (
        isinstance(module, nn.Linear)
        and (scope == "all" or ".mlp." in name)
        and min(module.weight.shape) >= 512
        and all(size % 16 == 0 for size in module.weight.shape)
    )


RECIPES = ("rowwise", "tensorwise", "rowwise_with_gw_hp", "rowwise_no_pow2")


def configure(model, recipe="rowwise", scope="all"):
    if scope not in ("all", "mlp"):
        raise ValueError("FP8 scope must be all or mlp")
    if recipe not in RECIPES:
        raise ValueError(f"Choose a supported FP8 recipe: {RECIPES}")
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    names = [name for name, module in model.named_modules() if eligible(module, name, scope)]
    before = {name: id(p) for name, p in model.named_parameters()}
    config = Float8LinearConfig.from_recipe_name(
        "rowwise" if recipe == "rowwise_no_pow2" else recipe
    )
    if recipe == "rowwise_no_pow2":
        config = dataclasses.replace(config, round_scales_to_power_of_2=False)
    convert_to_float8_training(
        model, module_filter_fn=lambda module, name: eligible(module, name, scope), config=config
    )
    assert before == {name: id(p) for name, p in model.named_parameters()}
    if not names:
        raise ValueError("No eligible FP8 linear layers")
    return names
