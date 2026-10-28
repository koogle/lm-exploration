"""Qualify the value-convolution-only RoPE candidate before matched training."""

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
def qualify_rope_v(run, deadline):
    import copy

    import model_hybrid.kernels.rope as rope_module
    import torch
    from model_hybrid.model import DocumentMask, HybridConfig, HybridLM, apply_rope
    from model_hybrid.study_mixers import configure

    assert time.time() < deadline - 300
    torch.set_num_threads(4)
    torch.manual_seed(17)
    base = HybridLM(HybridConfig(**CONFIG_GLM))
    candidate = configure(copy.deepcopy(base), "rope_v_conv")
    assert all(
        torch.equal(p, dict(base.named_parameters())[name])
        for name, p in candidate.named_parameters()
    )
    params = sum(p.numel() for p in candidate.parameters())
    for block in candidate.blocks:
        if block.kind == "kda":
            assert hasattr(block.mixer, "v_conv")
            assert not hasattr(block.mixer, "q_conv") and not hasattr(block.mixer, "k_conv")
    layer = candidate.blocks[0].mixer.cuda()
    reference = copy.deepcopy(layer)
    del base
    documents = (torch.arange(17, device="cuda") // 8).repeat(2, 1)
    mask = DocumentMask(documents)
    rope = (
        candidate.rope_cos.cuda()[mask.positions][:, None],
        candidate.rope_sin.cuda()[mask.positions][:, None],
    )
    x = torch.randn(2, 17, 448, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = layer(x, rope, mask)
    original = rope_module.rope_pair

    def reference_rope(q, k, cos, sin):
        return tuple(apply_rope(t.transpose(1, 2), cos, sin).transpose(1, 2) for t in (q, k))

    rope_module.rope_pair = reference_rope
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yr = reference(xr, rope, mask)
    finally:
        rope_module.rope_pair = original
    upstream = torch.randn_like(y)
    ga = torch.autograd.grad(y, (x, *layer.parameters()), upstream)
    gb = torch.autograd.grad(yr, (xr, *reference.parameters()), upstream)
    errors = []
    for a, b in zip([y, *ga], [yr, *gb]):
        absolute = (a.float() - b.float()).abs().max().item()
        relative = ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)).item()
        assert relative < 0.003 or absolute < 2e-6, (relative, absolute)
        errors.append({"max_abs": absolute, "relative_l2": relative})
    assert ga[1 + list(dict(layer.named_parameters())).index("v_conv")].abs().sum().item() > 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        changed = x.detach().clone()
        changed[:, :8] += 5
        after = layer(changed, rope, mask)
        assert torch.equal(y[:, 8:], after[:, 8:]), "Document leakage"
    result = {
        "variant": "rope_v_conv",
        "parameters": params,
        "errors": errors,
        "common_initialization_equal": True,
        "document_reset_passed": True,
        "v_convolution_receives_gradient": True,
        "recurrence_input_dtype": "bfloat16",
    }
    folder = Path("/runs") / run
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "qualification.json").write_text(json.dumps(result))
    runs.commit()
    result["data"] = preflight.local(run, deadline)
    print("QUALIFIED rope_v_conv", params, flush=True)
    return result
