"""Scalar-decay recurrence checks against expanded-channel KDA reference."""

import copy
import json

import torch
from model_hybrid import model
from model_hybrid.documents import document_ids, document_positions
from model_hybrid.gated_delta import configure, scalar_kernel, scalar_reference


def errors(actual, expected):
    result = []
    for a, b in zip(actual, expected, strict=True):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        value = ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-8)).item()
        assert value < 0.03, value
        result.append(value)
    return result


def kernel_check(production=False):
    torch.manual_seed(43)
    b, t, h, d = 2, 257, 12, 128
    tokens = torch.full((b, t), 3, device="cuda")
    tokens[:, [0, 64, 129, 255]] = 2
    documents = document_ids(tokens, 2)
    positions = document_positions(documents)
    leaves = [torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    leaves += [
        torch.randn(b, t, h, 1, device="cuda"),
        torch.randn(b, t, h, device="cuda"),
        torch.zeros(h, device="cuda"),
        torch.full((h,), -3.0, device="cuda"),
    ]
    if production:
        leaves[3], leaves[4] = leaves[3].bfloat16(), leaves[4].bfloat16()
        leaves[5] = torch.linspace(1, 16, h, device="cuda").log()
        dt = torch.logspace(-3, -1, h, device="cuda")
        leaves[6] = dt + torch.log(-torch.expm1(-dt))
    leaves = [x.requires_grad_() for x in leaves]
    out = scalar_kernel(*leaves, documents)
    dout = torch.randn_like(out)
    actual = (out, *torch.autograd.grad((out * dout).sum(), leaves))
    refs = [x.detach().float().requires_grad_() for x in leaves]
    ref = scalar_reference(*refs, positions)
    expected = (ref, *torch.autograd.grad((ref * dout.float()).sum(), refs))
    return errors(actual, expected)


def checks():
    result = {"kernel_errors": kernel_check(), "production_gate_errors": kernel_check(True)}
    print(json.dumps(result), flush=True)
    config = model.HybridConfig(
        vocab_size=521,
        eos_id=2,
        block_size=512,
        n_layer=2,
        n_embd=256,
        n_head=2,
        n_kv_head=1,
        head_dim=128,
        n_kda_head=2,
        mlp_hidden=512,
        layer_pattern=("kda", "full"),
    )
    lm = model.HybridLM(config).cuda()
    configure(lm)
    tokens = torch.randint(3, 521, (2, 257), device="cuda")
    tokens[:, 127] = 2
    labels = tokens.roll(-1, 1)
    labels[:, -1] = -100

    def run(net):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = net(tokens, labels, labels)
            loss = (r["loss"] + 0.3 * r["mtp_loss"]) / 512
        return (loss, *torch.autograd.grad(loss, tuple(net.parameters())))

    expected = run(lm)
    candidate = copy.deepcopy(lm)
    result["compiled_model_errors"] = errors(run(torch.compile(candidate)), expected)
    return result
