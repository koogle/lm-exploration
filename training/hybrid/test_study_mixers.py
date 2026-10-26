"""Study checks: scan recurrence, gradients, causality and packed documents."""


def checks():
    import torch

    from .model import HybridConfig, HybridLM
    from .study_mixers import RotatingMemory, configure

    torch.manual_seed(5)
    m = RotatingMemory(8)
    u = torch.randn(2, 17, 8, requires_grad=True)
    positions = torch.tensor([list(range(7)) + list(range(10))] * 2)
    actual = m(u, positions)
    z = torch.view_as_complex(u.reshape(2, 17, 4, 2))
    rho = torch.exp(-torch.exp(-m.log_tau))
    a = torch.polar(rho, m.theta)
    state = torch.zeros(2, 4, 4, dtype=torch.complex64)
    ys = []
    for t in range(17):
        state = state * (positions[:, t] > 0)[:, None, None]
        state = state * a[None, :, None] + z[:, t, None, :] * (1 - rho)[None, :, None]
        ys.append(u[:, t] + (torch.view_as_real(state).flatten(-2) * m.mix).sum(1))
    expected = torch.stack(ys, 1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    params = (u, *m.parameters())
    ga = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), params)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-5)
    cfg = HybridConfig(
        vocab_size=32,
        eos_id=0,
        n_layer=4,
        n_embd=32,
        n_head=2,
        n_kv_head=1,
        n_kda_head=2,
        head_dim=16,
        mlp_hidden=64,
        block_size=32,
    )
    for variant in ["baseline", "none", "rope", "middle_rope", "middle_rope_conv", "rotating"]:
        model = configure(HybridLM(cfg), variant)
        tokens = torch.tensor([[1, 2, 3, 0, 5, 6, 7, 8]])
        out = model(tokens)
        torch.testing.assert_close(out[:, 4:], model(tokens[:, 4:]), atol=1e-6, rtol=1e-5)
        changed = tokens.clone()
        changed[:, -1] = 9
        torch.testing.assert_close(out[:, :-1], model(changed)[:, :-1], atol=1e-6, rtol=1e-5)
        out.square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return {
        "scan_forward_and_backward": "passed",
        "document_isolation": "passed",
        "causality": "passed",
        "parameter_gradients": "passed",
    }
