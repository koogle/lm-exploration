"""Benchmark-only Hopper FP8 linears; recurrence and vocabulary loss stay unchanged."""

from torch import nn


def eligible(module, name):
    # Retain small gate/normalization-related maps in BF16. Cover the production
    # MLPs and large attention projections, including MTP's full-attention block.
    return (
        isinstance(module, nn.Linear)
        and min(module.weight.shape) >= 512
        and all(size % 16 == 0 for size in module.weight.shape)
    )


RECIPES = ("rowwise", "tensorwise", "rowwise_with_gw_hp")


def configure(model, recipe="rowwise"):
    if recipe not in RECIPES:
        raise ValueError(f"Choose a supported FP8 recipe: {RECIPES}")
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    names = [name for name, module in model.named_modules() if eligible(module, name)]
    before = {name: id(p) for name, p in model.named_parameters()}
    convert_to_float8_training(
        model, module_filter_fn=eligible, config=Float8LinearConfig.from_recipe_name(recipe)
    )
    assert before == {name: id(p) for name, p in model.named_parameters()}
    if not names:
        raise ValueError("No eligible FP8 linear layers")
    return names
