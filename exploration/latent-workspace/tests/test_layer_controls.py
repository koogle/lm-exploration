from dataclasses import replace

import torch
from test_core import tiny_config

from latent_workspace.models import Workspace


def test_null_all_dropped_is_finite_and_independent_of_memory():
    cfg = replace(tiny_config(), layer_dropout=1.0, null_attention=True)
    workspace = Workspace(16, cfg).train()
    memory = torch.randn(3, len(cfg.layers), cfg.memory_tokens, 16)
    a = workspace(memory)
    b = workspace(memory * 100)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b)
    a.sum().backward()
    assert workspace.project.weight.grad.abs().max() == 0


def test_dropout_has_independent_rng_and_is_disabled_at_eval():
    cfg = replace(tiny_config(), layer_dropout=0.5)
    workspace = Workspace(16, cfg)
    memory = torch.randn(3, len(cfg.layers), cfg.memory_tokens, 16)
    state = torch.get_rng_state()
    workspace(memory)
    assert torch.equal(state, torch.get_rng_state())
    workspace.eval()
    a = workspace(memory)
    workspace.dropout_step = 100
    torch.testing.assert_close(a, workspace(memory))


def test_null_control_preserves_parameter_initialization():
    cfg = tiny_config()
    torch.manual_seed(42)
    a = Workspace(16, cfg)
    torch.manual_seed(42)
    b = Workspace(16, replace(cfg, null_attention=True))
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key])
