"""CUDA checks for the hybrid family: our attention kernel and fla's KDA kernel.

The attention kernel must be bitwise identical to the fixed-order definition in
exact_attention.py (forward outputs and statistics, all input gradients), give the
same bits on repeated runs, stay close to fp64 softmax attention, and work inside
torch.compile. fla's chunked KDA must match kda_reference within TF32 tolerance,
and a small hybrid model is timed with fla inside and outside the compiled graph.
Run on one GPU through model_hybrid.modal_checks; returns a
JSON-serializable dict and raises on any mismatch.
"""

import time

import torch
from model_hybrid import exact_attention, kda, model
from model_hybrid.document_attention import (
    _backward,
    _forward,
    document_attention,
    softmax_scale,
)
from model_hybrid.documents import document_ids, document_positions

EOS = 2


def packed(batch, length, ends):
    tokens = torch.randint(3, 100, (batch, length), device="cuda")
    for row, positions in enumerate(ends):
        tokens[row, positions] = EOS
    return exact_attention.document_starts(document_ids(tokens, EOS))


def max_error(actual, expected):
    return (actual.double() - expected.double()).abs().max().item()


def bitwise(name, actual, expected):
    if actual.dtype != expected.dtype or not torch.equal(actual, expected):
        differing = (actual != expected).sum().item()
        raise AssertionError(
            f"{name}: {differing} values differ, max {max_error(actual, expected)}"
        )


def against_reference(dtype, batch, heads, kv_heads, length, width, ends):
    torch.manual_seed(length + width)
    q = torch.randn(batch, heads, length, width, device="cuda").to(dtype)
    k = torch.randn(batch, kv_heads, length, width, device="cuda").to(dtype)
    v = torch.randn(batch, kv_heads, length, width, device="cuda").to(dtype)
    starts = packed(batch, length, ends)
    scale = softmax_scale(width)
    kernel = _forward(q, k, v, starts, scale)
    reference = exact_attention.exact_attention(q, k, v, starts, scale)
    for name, a, b in zip(("out", "lse", "max", "sum"), kernel, reference, strict=True):
        bitwise(name, a, b)
    bitwise("repeat forward", _forward(q, k, v, starts, scale)[0], kernel[0])

    dout = torch.randn_like(q)
    dlse = torch.randn(batch, heads, length, device="cuda")
    out, _, row_max, row_sum = kernel
    grads = _backward(q, k, v, starts, out, dout, dlse, row_max, row_sum, scale)
    expected = exact_attention.exact_attention_backward(
        q, k, v, starts, scale, out, dout, dlse, row_max, row_sum
    )
    for name, a, b in zip(("dq", "dk", "dv"), grads, expected, strict=True):
        bitwise(name, a, b)
    repeat = _backward(q, k, v, starts, out, dout, dlse, row_max, row_sum, scale)
    for name, a, b in zip(("dq", "dk", "dv"), repeat, grads, strict=True):
        bitwise(f"repeat {name}", a, b)

    # Sanity against fp64 softmax attention and autograd.
    leaves = [t.double().requires_grad_() for t in (q, k, v)]
    index = torch.arange(length, device="cuda")
    allowed = (index[None, :] <= index[:, None]) & (index[None, :] >= starts[:, None, :, None])
    fp64, fp64_lse = model.reference_attention(*leaves, allowed)
    objective = (fp64 * dout.double()).sum() + (fp64_lse * dlse.double()).sum()
    fp64_grads = torch.autograd.grad(objective, leaves)
    return {
        "shape": [batch, heads, kv_heads, length, width],
        "bitwise_equal": True,
        "vs_fp64": {
            "out": max_error(out, fp64),
            "lse": max_error(kernel[1], fp64_lse),
            **{f"d{n}": max_error(a, b) for n, a, b in zip("qkv", grads, fp64_grads, strict=True)},
        },
    }


def flex_against_exact(length=1024, ends=((100, 400, 401, 900), (511,))):
    """The training path (FlexAttention) against our exact kernel, bf16 inputs."""
    torch.manual_seed(5)
    q = torch.randn(2, 12, length, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 4, length, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    tokens = torch.randint(3, 100, (2, length), device="cuda")
    for row, positions in enumerate(ends):
        tokens[row, list(positions)] = EOS
    documents = document_ids(tokens, EOS)
    upstream = torch.randn_like(q)
    upstream_lse = torch.randn(2, 12, length, device="cuda")
    results = {}
    for kernel in ("flex", "exact"):
        out, lse = model.attend(q, k, v, model.DocumentMask(documents, kernel))
        objective = (out.float() * upstream).sum() + (lse.float() * upstream_lse).sum()
        results[kernel] = (out, lse, *torch.autograd.grad(objective, (q, k, v)))
    names = ("out", "lse", "dq", "dk", "dv")
    errors = {
        n: max_error(a, b) for n, a, b in zip(names, results["flex"], results["exact"], strict=True)
    }
    if max(errors.values()) > 0.1:
        raise AssertionError(f"FlexAttention disagrees with the exact kernel: {errors}")
    return errors


def softmax_gradient_against_reference():
    """Our CUDA softmax-gradient kernel against fp64 PyTorch, aligned and unaligned rows."""
    from model_hybrid.softmax_kernel import softmax_gradient_

    results = {}
    for dtype, vocab in (
        (torch.float32, 5001),
        (torch.float32, 154_880),
        (torch.bfloat16, 154_880),
    ):
        torch.manual_seed(vocab)
        logits = (torch.randn(64, vocab, device="cuda") * 4).to(dtype)
        targets = torch.randint(0, vocab, (64,), device="cuda")
        targets[::5] = -100
        reference = logits.double()
        lse = reference.logsumexp(-1)
        expected = (reference - lse[:, None]).exp() * (1 + 2 * 1e-4 * lse)[:, None]
        rows = torch.arange(64, device="cuda")
        valid = targets != -100
        expected[rows[valid], targets[valid]] -= 1
        expected[~valid] = 0
        actual = logits.clone()
        got_lse, got_target = softmax_gradient_(actual, targets, 1e-4)
        errors = {
            "lse": max_error(got_lse, lse),
            "target_logit": max_error(got_target, reference[rows, targets.clamp_min(0)]),
            "gradient": max_error(actual, expected),
        }
        limit = 1e-5 if dtype == torch.float32 else 4e-3
        if errors["lse"] > 1e-4 or errors["target_logit"] > 0 or errors["gradient"] > limit:
            raise AssertionError(f"softmax gradient kernel mismatch ({dtype}, {vocab}): {errors}")
        results[f"{str(dtype).split('.')[-1]}_{vocab}"] = errors
    return results


def fused_loss_against_autograd():
    """Fused output-layer loss (CUDA gradient kernel) against the chunked autograd loss."""
    from model_hybrid.fused_loss import fused_cross_entropy

    torch.manual_seed(6)
    hidden = torch.randn(4096, 1536, device="cuda", requires_grad=True)
    weight = (torch.randn(154_880, 1536, device="cuda") * 0.02).requires_grad_()
    labels = torch.randint(0, 154_856, (4096,), device="cuda")
    labels[::7] = -100
    loss, ce, z = fused_cross_entropy(hidden, weight, labels, 1e-4, 2048)
    fused = torch.autograd.grad(loss, (hidden, weight))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected_ce, expected_z = model.chunked_cross_entropy(hidden, weight, labels, 2048)
    reference = torch.autograd.grad(expected_ce + 1e-4 * expected_z, (hidden, weight))
    errors = {
        "ce_relative": abs(ce.item() - expected_ce.item()) / expected_ce.item(),
        "z_relative": abs(z.item() - expected_z.item()) / expected_z.item(),
        **{
            f"grad_{n}_relative": max_error(a, b) / b.abs().max().item()
            for n, a, b in zip(("hidden", "weight"), fused, reference, strict=True)
        },
    }
    if max(errors.values()) > 2e-2:
        raise AssertionError(f"Fused loss disagrees with autograd: {errors}")
    return errors


def compiled_model():
    torch.manual_seed(1)
    # Full attention only: fullgraph proves the kernel's custom ops do not break the
    # graph (the fla call in KDA layers is a deliberate graph break).
    config = model.HybridConfig(
        vocab_size=512,
        block_size=256,
        n_layer=2,
        n_embd=256,
        n_head=4,
        n_kv_head=2,
        head_dim=64,
        layer_pattern=("full",),
        mtp=False,
    )
    lm = model.HybridLM(config).cuda()
    compiled = torch.compile(lm, fullgraph=True)
    b = torch.randint(3, 512, (1, 100), device="cuda")
    first = torch.cat((torch.randint(3, 512, (1, 155), device="cuda"), b), 1)
    first[0, 154] = EOS
    second = torch.cat((torch.randint(3, 512, (1, 99), device="cuda"), b), 1)
    second[0, 98] = EOS
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = compiled(first, documents=document_ids(first, EOS))[0, 155:]
        c = compiled(second, documents=document_ids(second, EOS))[0, 99:]
        labels = torch.randint(0, 512, first.shape, device="cuda")
        losses = compiled(first, labels, document_ids(first, EOS))
    losses["loss"].backward()
    if not all(torch.isfinite(p.grad).all() for p in lm.parameters()):
        raise AssertionError("Non-finite gradients in the compiled model")
    return {"isolation_max_logit_error_bf16": max_error(a, c), "loss": losses["loss"].item()}


def timing(batch=8, heads=12, kv_heads=4, length=4096, width=128, documents=1):
    """Forward and forward+backward milliseconds, ours against PyTorch's causal SDPA."""
    q = torch.randn(batch, heads, length, width, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, length, width, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ends = [list(range(length // documents - 1, length - 1, length // documents))] * batch
    starts = packed(batch, length, ends)
    leaves = [t.requires_grad_() for t in (q, k, v)]

    def ours():
        return document_attention(*leaves, starts, softmax_scale(width))[0]

    def sdpa():
        return torch.nn.functional.scaled_dot_product_attention(
            *leaves, is_causal=True, enable_gqa=True
        )

    result = {"shape": [batch, heads, kv_heads, length, width], "documents": documents}
    for name, function in (("ours", ours), ("sdpa_causal_one_document", sdpa)):
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


def kda_against_reference(dtype, tolerance):
    torch.manual_seed(2)
    batch, length, heads, width = 2, 512, 4, 128

    def leaf(*shape):
        return torch.randn(*shape, device="cuda").to(dtype).requires_grad_()

    q, k, v = (leaf(batch, length, heads, width) for _ in range(3))
    g, beta = leaf(batch, length, heads, width), leaf(batch, length, heads)
    A_log = torch.empty(heads, device="cuda").uniform_(1, 16).log().requires_grad_()
    dt_bias = (torch.randn(heads * width, device="cuda") - 4).requires_grad_()
    tokens = torch.randint(3, 100, (batch, length), device="cuda")
    tokens[0, [63, 64, 300]] = EOS
    tokens[1, 200] = EOS
    documents = document_ids(tokens, EOS)
    inputs = (q, k, v, g, beta, A_log, dt_bias)
    upstream = torch.randn(batch, length, heads, width, device="cuda")

    def run(function, last):
        output = function(*inputs, last)
        return output, torch.autograd.grad((output.float() * upstream).sum(), inputs)

    fla_out, fla_grads = run(kda.kda_kernel, documents)
    ref_out, ref_grads = run(kda.kda_reference, document_positions(documents))
    names = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
    errors = {
        "output": max_error(fla_out, ref_out) / ref_out.float().abs().max().item(),
        **{
            f"grad_{name}": max_error(a, b) / max(b.float().abs().max().item(), 1e-6)
            for name, a, b in zip(names, fla_grads, ref_grads, strict=True)
        },
    }
    if max(errors.values()) > tolerance:
        raise AssertionError(f"KDA kernel mismatch for {dtype}: {errors}")
    return {"relative_max_error": errors}


def compile_boundary_timing(rounds=3, steps=10):
    """bf16 training-step time of a small compiled hybrid model, fla inside or outside.

    Both variants compile and warm up (including fla's autotuning) before any
    timing; timing rounds then alternate between them.
    """
    torch.manual_seed(3)
    config = model.HybridConfig(
        vocab_size=4096,
        block_size=1024,
        n_layer=4,
        n_embd=512,
        n_head=4,
        n_kv_head=2,
        n_kda_head=4,
        mlp_hidden=1024,
    )
    lm = model.HybridLM(config).cuda()
    tokens = torch.randint(3, 4096, (4, 1024), device="cuda")
    tokens[:, 300] = EOS
    labels = torch.randint(0, 4096, tokens.shape, device="cuda")
    documents = document_ids(tokens, EOS)

    def variant(function):
        # Both compiled variants share one code cache, guarded on the kernel function.
        kda.kda_kernel = function
        compiled = torch.compile(lm)

        def step():
            kda.kda_kernel = function
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = compiled(tokens, labels, documents)
            losses["loss"].backward()

        for _ in range(5):
            step()
        return step

    original = kda.kda_kernel
    try:
        steps_by_name = {
            "fla_outside_graph": variant(torch.compiler.disable(original)),
            "fla_inside_graph": variant(original),
        }
        times = {name: [] for name in steps_by_name}
        for _ in range(rounds):
            for name, step in steps_by_name.items():
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(steps):
                    step()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - start) * 1000 / steps)
    finally:
        kda.kda_kernel = original
    return {f"{name}_step_ms": values for name, values in times.items()}


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
        "kernel_vs_exact_reference": [
            {"dtype": str(case[0]), **against_reference(*case)} for case in cases
        ],
        # fla's Triton dots use TF32 even for fp32 inputs.
        "kda_vs_reference_fp32": kda_against_reference(torch.float32, 1e-2),
        "kda_vs_reference_bf16": kda_against_reference(torch.bfloat16, 5e-2),
        "flex_vs_exact_bf16": flex_against_exact(),
        "softmax_gradient_vs_fp64": softmax_gradient_against_reference(),
        "fused_loss_vs_autograd": fused_loss_against_autograd(),
        "compiled_model": compiled_model(),
        "compile_boundary": compile_boundary_timing(),
        "timing": [timing(documents=1), timing(documents=4)],
    }
