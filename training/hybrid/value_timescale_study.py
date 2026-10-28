"""GPU correctness gate for local convolution plus learned value timescales."""

import json
import time
from pathlib import Path

from .branch_study import CONFIG_GLM, preflight, prepared_volume
from .modal_study import app, image, runs


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=900,
    retries=0,
    volumes={"/data": prepared_volume, "/runs": runs},
)
def qualify_value_timescales(run, deadline):
    import copy

    import torch
    from model_hybrid.model import DocumentMask, HybridConfig, HybridLM
    from model_hybrid.study_mixers import configure
    from model_hybrid.value_timescales import ValueTimescales

    assert time.time() < deadline - 300
    torch.set_num_threads(4)
    torch.manual_seed(17)
    checks = []
    for length in (17, 1024):
        for dtype in (torch.float32, torch.bfloat16):
            mixer = ValueTimescales(64).cuda()
            with torch.no_grad():
                mixer.mix.normal_(0, 0.2)
            ref = copy.deepcopy(mixer)
            ref.use_kernel = False
            u = torch.randn(2, length, 64, device="cuda", dtype=dtype, requires_grad=True)
            ur = u.detach().clone().requires_grad_()
            w = torch.randn(64, 1, 4, device="cuda", requires_grad=True)
            wr = w.detach().clone().requires_grad_()
            positions = (torch.arange(length, device="cuda") % 73).repeat(2, 1)
            y, yr = mixer(u, w, positions), ref(ur, wr, positions)
            upstream = torch.randn_like(y)
            ga = torch.autograd.grad(y, (u, w, *mixer.parameters()), upstream)
            gb = torch.autograd.grad(yr, (ur, wr, *ref.parameters()), upstream)
            errors = []
            for a, b in zip((y, *ga), (yr, *gb)):
                relative = (
                    (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
                ).item()
                absolute = (a.float() - b.float()).abs().max().item()
                assert relative < 0.005 or absolute < 2e-5, (length, dtype, relative, absolute)
                errors.append({"relative_l2": relative, "max_abs": absolute})
            with torch.no_grad():
                changed = u.clone()
                changed[:, : min(73, length)] += 10
                if length > 73:
                    assert torch.equal(mixer(changed, w, positions)[:, 73:], y[:, 73:])
                changed = u.clone()
                changed[:, length // 2 :] += 10
                assert torch.equal(
                    mixer(changed, w, positions)[:, : length // 2], y[:, : length // 2]
                )
            checks.append({"length": length, "dtype": str(dtype), "errors": errors})
    base = HybridLM(HybridConfig(**CONFIG_GLM))
    control = configure(copy.deepcopy(base), "rope_v_conv")
    candidate = configure(base, "rope_v_timescales")
    cp = dict(control.named_parameters())
    assert all(torch.equal(p, cp[n]) for n, p in candidate.named_parameters() if n in cp)
    layer, local = candidate.blocks[0].mixer.cuda(), control.blocks[0].mixer.cuda()
    documents = (torch.arange(17, device="cuda") // 8).repeat(2, 1)
    mask = DocumentMask(documents)
    rope = (
        candidate.rope_cos.cuda()[mask.positions][:, None],
        candidate.rope_sin.cuda()[mask.positions][:, None],
    )
    x = torch.randn(2, 17, 448, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, yc = layer(x, rope, mask), local(x, rope, mask)
    assert torch.equal(y, yc), "Zero branch must preserve initial function"
    y.float().square().mean().backward()
    assert layer.value_timescales.mix.grad.abs().sum() > 0
    # Zero mix deliberately blocks tau gradients initially; after opening the
    # branch, both the mixture and timescale derivatives must be nonzero.
    assert all(g.abs().sum() > 0 for g in ga)
    result = {
        "variant": "rope_v_timescales",
        "checks": checks,
        "parameters": sum(p.numel() for p in candidate.parameters()),
        "initial_function_equal": True,
        "shared_parameters_equal": True,
        "causality_and_document_reset": True,
        "mixture_gradient_nonzero": True,
    }
    folder = Path("/runs") / run
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "qualification.json").write_text(json.dumps(result))
    runs.commit()
    result["data"] = preflight.local(run, deadline)
    print("QUALIFIED rope_v_timescales", result["parameters"], flush=True)
    return result
