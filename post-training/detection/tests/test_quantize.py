import pytest
import torch

from parallel_detection.quantize import round_weights_


def linears(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.Linear)]


@pytest.mark.parametrize("bits", [4, 8])
def test_rounding_error_is_at_most_half_a_step(bits):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(70, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    before = [m.weight.detach().clone() for m in linears(model)]
    assert round_weights_(model, bits, group=32) == (2, 70 * 16 + 16 * 4)
    top = 2 ** (bits - 1) - 1
    for old, layer in zip(before, linears(model), strict=True):
        for start in range(0, old.shape[1], 32):
            block = old[:, start : start + 32]
            step = block.abs().amax(1, keepdim=True) / top
            error = (layer.weight[:, start : start + 32] - block).abs()
            assert (error <= step / 2 + 1e-6).all()


def test_each_group_takes_at_most_two_to_the_bits_values():
    torch.manual_seed(1)
    model = torch.nn.Linear(32, 3)
    round_weights_(model, 4, group=32)
    for row in model.weight:
        assert len(torch.unique(row)) <= 16


def test_a_module_without_linear_layers_is_refused():
    with pytest.raises(ValueError):
        round_weights_(torch.nn.ReLU(), 4)
