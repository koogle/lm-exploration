"""CUDA checks for the hybrid family's attention.

Compiled FlexAttention with the document-causal block mask must stay close to fp64
softmax attention (outputs, log-sum-exp and all input gradients, including through
the log-sum-exp) and work inside torch.compile. Run on one GPU through
model_hybrid.modal_checks; returns a JSON-serializable dict and raises on any
mismatch.
"""

import time

import torch
from model_hybrid import model
from model_hybrid.documents import document_ids

EOS = 2
# bf16 rounding of attention outputs and gradients of unit-scale inputs.
TOLERANCE = {torch.float32: 2e-3, torch.bfloat16: 0.1}


def packed(batch, length, ends):
    tokens = torch.randint(3, 100, (batch, length), device="cuda")
    for row, positions in enumerate(ends):
        tokens[row, positions] = EOS
    return model.DocumentMask(document_ids(tokens, EOS))


def max_error(actual, expected):
    return (actual.double() - expected.double()).abs().max().item()


def against_reference(dtype, batch, heads, kv_heads, length, width, ends):
    torch.manual_seed(length + width)
    tensors = [
        torch.randn(batch, h, length, width, device="cuda").to(dtype).requires_grad_()
        for h in (heads, kv_heads, kv_heads)
    ]
    mask = packed(batch, length, ends)
    out, lse = model.attend(*tensors, mask)
    dout = torch.randn_like(out)
    dlse = torch.randn_like(lse)
    grads = torch.autograd.grad((out * dout).sum() + (lse * dlse).sum(), tensors)
    leaves = [t.detach().double().requires_grad_() for t in tensors]
    fp64, fp64_lse = model.reference_attention(*leaves, mask.dense())
    objective = (fp64 * dout.double()).sum() + (fp64_lse * dlse.double()).sum()
    fp64_grads = torch.autograd.grad(objective, leaves)
    errors = {
        "out": max_error(out, fp64),
        "lse": max_error(lse, fp64_lse),
        **{f"d{n}": max_error(a, b) for n, a, b in zip("qkv", grads, fp64_grads, strict=True)},
    }
    if max(errors.values()) > TOLERANCE[dtype]:
        raise AssertionError(f"FlexAttention differs from fp64 attention: {errors}")
    return {"shape": [batch, heads, kv_heads, length, width], "vs_fp64": errors}


def compiled_model():
    torch.manual_seed(1)
    config = model.HybridConfig(
        vocab_size=512,
        eos_id=EOS,
        block_size=256,
        n_layer=2,
        n_embd=256,
        n_head=4,
        n_kv_head=2,
        head_dim=64,
    )
    lm = model.HybridLM(config).cuda()
    compiled = torch.compile(lm)
    b = torch.randint(3, 512, (1, 100), device="cuda")
    first = torch.cat((torch.randint(3, 512, (1, 155), device="cuda"), b), 1)
    first[0, 154] = EOS
    second = torch.cat((torch.randint(3, 512, (1, 99), device="cuda"), b), 1)
    second[0, 98] = EOS
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = compiled(first)[0, 155:]
        c = compiled(second)[0, 99:]
        labels = torch.randint(0, 512, first.shape, device="cuda")
        losses = compiled(first, labels)
    losses["loss"].backward()
    if not all(torch.isfinite(p.grad).all() for p in lm.parameters()):
        raise AssertionError("Non-finite gradients in the compiled model")
    return {"isolation_max_logit_error_bf16": max_error(a, c), "loss": losses["loss"].item()}


def timing(batch=8, heads=12, kv_heads=4, length=4096, width=128, documents=1):
    """Forward and forward+backward milliseconds: FlexAttention vs PyTorch causal SDPA."""
    q = torch.randn(batch, heads, length, width, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, length, width, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ends = [list(range(length // documents - 1, length - 1, length // documents))] * batch
    mask = packed(batch, length, ends)
    leaves = [t.requires_grad_() for t in (q, k, v)]

    def flex():
        return model.attend(*leaves, mask)[0]

    def sdpa():
        return torch.nn.functional.scaled_dot_product_attention(
            *leaves, is_causal=True, enable_gqa=True
        )

    result = {"shape": [batch, heads, kv_heads, length, width], "documents": documents}
    for name, function in (("flex", flex), ("sdpa_causal_one_document", sdpa)):
        for mode in ("forward", "forward_backward"):

            def step(function=function, mode=mode):
                out = function()
                if mode == "forward_backward":
                    out.backward(torch.ones_like(out))

            for _ in range(2):
                step()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(5):
                step()
            torch.cuda.synchronize()
            result[f"{name}_{mode}_ms"] = (time.perf_counter() - start) * 200
    return result


def checks():
    cases = (
        (torch.float32, 2, 12, 4, 300, 128, [[63, 64, 200], [150]]),
        (torch.bfloat16, 2, 12, 4, 300, 128, [[63, 64, 200], [150]]),
        (torch.bfloat16, 1, 4, 4, 129, 64, [[0, 70]]),
        (torch.bfloat16, 1, 8, 2, 1024, 128, [[]]),
    )
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "flex_vs_fp64": [{"dtype": str(case[0]), **against_reference(*case)} for case in cases],
        "compiled_model": compiled_model(),
        "timing": [timing(documents=1), timing(documents=4)],
    }
