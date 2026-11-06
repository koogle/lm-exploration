"""Timing and accuracy of one KDA layer's kernels at the 720M training shapes.

Run on one GPU through `modal run -m model_hybrid.modal_checks::kda_bench`. Times
forward and forward+backward of the recurrence (fla's chunk_kda, and any other
implementation passed in) and of the three short convolutions, with GPU time per
kernel from the PyTorch profiler, and compares outputs and gradients with
`kda_reference` on a shorter sequence. Returns a JSON-serializable dict.
"""

import collections
import time

import torch

from . import kda
from .documents import document_ids, document_positions

EOS = 2
SHAPE = {"batch": 4, "length": 4096, "heads": 12, "width": 128}


def packed_documents(batch, length, mean, generator):
    """Document ids with geometric lengths of the given mean, as in the recipe."""
    tokens = torch.full((batch, length), 5)
    ends = torch.rand(batch, length, generator=generator) < 1 / mean
    tokens[ends] = EOS
    return document_ids(tokens, EOS).cuda()


def layer_inputs(batch, length, heads, width, documents, dtype, generator):
    """Raw kernel inputs as KimiDeltaAttention passes them, with leaf gradients."""
    shape = (batch, length, heads, width)

    def leaf(*size, scale=1.0):
        return (scale * torch.randn(*size, generator=generator)).cuda().to(dtype)

    q, k, v = (torch.nn.functional.silu(leaf(*shape)) for _ in range(3))
    g, beta = leaf(*shape, scale=0.5), leaf(batch, length, heads)
    A_log = torch.empty(heads).uniform_(1, 16, generator=generator).log().cuda()
    dt = torch.exp(torch.rand(heads * width, generator=generator) * 4.6 - 6.9)
    dt_bias = (dt + torch.log(-torch.expm1(-dt))).cuda()
    tensors = [t.detach().requires_grad_() for t in (q, k, v, g, beta, A_log, dt_bias)]
    return tensors, documents


def kernel_times(function, steps=5):
    """Wall ms per call and GPU ms per kernel name, after two warmup calls."""
    for _ in range(2):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        function()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) * 1000 / steps
    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profiler:
        for _ in range(steps):
            function()
        torch.cuda.synchronize()
    kernels = collections.Counter()
    for event in profiler.events():
        if event.device_type.name == "CUDA" and not event.is_user_annotation:
            kernels[event.name[:120]] += event.self_device_time_total / 1000 / steps
    return {
        "wall_ms": wall,
        "gpu_ms": sum(kernels.values()),
        "kernels_ms": dict(kernels.most_common(12)),
    }


def time_recurrence(implementation, dtype=torch.bfloat16, mean_document=1000):
    generator = torch.Generator().manual_seed(0)
    s = SHAPE
    documents = packed_documents(s["batch"], s["length"], mean_document, generator)
    tensors, documents = layer_inputs(
        s["batch"], s["length"], s["heads"], s["width"], documents, dtype, generator
    )
    upstream = torch.randn(s["batch"], s["length"], s["heads"], s["width"]).cuda().to(dtype)

    def forward():
        with torch.no_grad():
            return implementation(*tensors, documents)

    def forward_backward():
        out = implementation(*tensors, documents)
        torch.autograd.grad(out, tensors, upstream)

    return {"forward": kernel_times(forward), "forward_backward": kernel_times(forward_backward)}


def convolution_inputs(dtype, generator, batch=None, length=None, mean_document=1000):
    s = SHAPE
    batch, length = batch or s["batch"], length or s["length"]
    documents = packed_documents(batch, length, mean_document, generator)
    channels = s["heads"] * s["width"]
    xs = [
        torch.randn(batch, length, channels, generator=generator).cuda().to(dtype).requires_grad_()
        for _ in range(3)
    ]
    weights = [
        (0.5 * torch.randn(channels, 1, 4, generator=generator)).cuda().requires_grad_()
        for _ in range(3)
    ]
    return xs, weights, document_positions(documents)


def compiled_convolutions():
    convolution = torch.compile(kda.short_convolution)

    def run(xq, xk, xv, wq, wk, wv, positions):
        return tuple(
            convolution(x, w, positions) for x, w in zip((xq, xk, xv), (wq, wk, wv), strict=True)
        )

    return run


def time_convolutions(implementation, dtype=torch.bfloat16):
    """Forward+backward of one layer's three short convolutions."""
    xs, weights, positions = convolution_inputs(dtype, torch.Generator().manual_seed(1))

    def forward_backward():
        outs = implementation(*xs, *weights, positions)
        torch.autograd.grad(outs, [*xs, *weights], [torch.ones_like(o) for o in outs])

    return kernel_times(forward_backward)


def convolution_accuracy(implementation, dtype, tolerance):
    """Relative max error of outputs and x/w gradients against fp64 short_convolution."""
    generator = torch.Generator().manual_seed(3)
    xs, weights, positions = convolution_inputs(
        dtype, generator, batch=2, length=777, mean_document=40
    )
    outs = implementation(*xs, *weights, positions)
    upstream = [torch.randn(o.shape, generator=generator).cuda() for o in outs]
    grads = torch.autograd.grad(
        outs, [*xs, *weights], [u.to(o.dtype) for u, o in zip(upstream, outs)]
    )
    leaves = [t.detach().double().requires_grad_() for t in (*xs, *weights)]
    expected = [kda.short_convolution(x, w, positions) for x, w in zip(leaves[:3], leaves[3:])]
    expected_grads = torch.autograd.grad(expected, leaves, [u.double() for u in upstream])
    errors = {}
    names = ("q", "k", "v", "dq", "dk", "dv", "dwq", "dwk", "dwv")
    for name, a, b in zip(names, (*outs, *grads), (*expected, *expected_grads), strict=True):
        errors[name] = ((a.double() - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()
    if max(errors.values()) > tolerance:
        raise AssertionError(f"Convolutions differ from the reference: {errors}")
    return errors


def accuracy(implementation, dtype, length=512, tolerance=5e-2, width=64):
    """Relative max error of outputs and all input gradients against kda_reference."""
    generator = torch.Generator().manual_seed(2)
    documents = packed_documents(2, length, 150, generator)
    tensors, documents = layer_inputs(2, length, 4, width, documents, dtype, generator)
    upstream = torch.randn(2, length, 4, width, generator=generator).cuda()
    out = implementation(*tensors, documents)
    grads = torch.autograd.grad(out, tensors, upstream.to(out.dtype))
    reference_inputs = [t.detach().double().requires_grad_() for t in tensors]
    expected = kda.kda_reference(*reference_inputs, document_positions(documents))
    expected_grads = torch.autograd.grad(expected, reference_inputs, upstream.double())
    names = ("output", "q", "k", "v", "g", "beta", "A_log", "dt_bias")
    errors = {}
    for name, a, b in zip(names, (out, *grads), (expected, *expected_grads), strict=True):
        scale = b.abs().max().clamp_min(1e-12)
        errors[name] = ((a.double() - b).abs().max() / scale).item()
    if max(errors.values()) > tolerance:
        raise AssertionError(f"KDA differs from the reference: {errors}")
    return errors


def run(implementations=None):
    from .kda_convolution import kda_convolutions

    implementations = implementations or {"fla": kda.kda_kernel}
    convolutions = {"compiled_torch": compiled_convolutions(), "cuda_kernel": kda_convolutions}
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": SHAPE,
        "accuracy": {
            f"{name}_{str(dtype)[6:]}": accuracy(function, dtype)
            for name, function in implementations.items()
            for dtype in (torch.float32, torch.bfloat16)
        },
        "recurrence": {name: time_recurrence(f) for name, f in implementations.items()},
        "convolution_accuracy": {
            f"{name}_{str(dtype)[6:]}": convolution_accuracy(function, dtype, tolerance)
            for name, function in convolutions.items()
            for dtype, tolerance in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2))
        },
        "short_convolutions": {name: time_convolutions(f) for name, f in convolutions.items()},
    }


def wy_tiling(data_root="/data", hoisted=False):
    """Measure WY launch layouts on tensors produced from real 4K document packing."""
    import importlib
    import json
    from pathlib import Path

    import triton
    import triton.testing
    from training_data.packed_shards import PreparedMixture

    module = importlib.import_module("fla.ops.kda.chunk_bwd")
    original = module.chunk_kda_bwd_kernel_wy_dqkg_fused
    captured = {}

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                if args:
                    raise AssertionError("Expected named WY launch arguments")
                captured.update(grid=grid, kwargs=kwargs)
                return original[grid](**kwargs)

            return launch

    recipe = json.loads(
        (Path(__file__).parent / "recipes/kda-foundation-100b-504010-v1.json").read_text()
    )
    mixture = PreparedMixture(recipe, Path(data_root) / recipe["name"])
    mixture.wait_seconds = 0
    tokens, _, sources = mixture.next_batch(4)
    documents = document_ids(torch.as_tensor(tokens, device="cuda"), 154820)
    generator = torch.Generator().manual_seed(14)
    tensors, _ = layer_inputs(4, 4096, 12, 128, documents, torch.bfloat16, generator)
    module.chunk_kda_bwd_kernel_wy_dqkg_fused = Capture()
    try:
        out = kda.kda_kernel(*tensors, documents)
        torch.autograd.grad(out, tensors, torch.ones_like(out))
    finally:
        module.chunk_kda_bwd_kernel_wy_dqkg_fused = original
    grid, kwargs = captured["grid"], captured["kwargs"]
    output_names = ("dq", "dk", "dv2", "dg", "db", "dA")
    original[grid](**kwargs)
    expected = {name: kwargs[name].clone() for name in output_names}
    tuner = original.fn
    selected = tuner.best_config
    jit = tuner.fn
    configurations = [("upstream", selected)] + [
        (
            f"bk{bk}-bv{bv}-w{warps}-s{stages}",
            triton.Config({"BK": bk, "BV": bv}, num_warps=warps, num_stages=stages),
        )
        for bk, bv, warps, stages in (
            (32, 64, 4, 1),
            (32, 64, 4, 2),
            (32, 64, 8, 1),
            (32, 32, 4, 1),
            (16, 64, 4, 1),
            (16, 32, 4, 1),
            (64, 64, 8, 1),
            (64, 64, 8, 2),
            (32, 128, 8, 1),
        )
    ]
    if hoisted:
        from .kernels.kda_wy import wy_backward

        configurations = [configurations[0]] + [
            ("hoisted-" + name, config) for name, config in configurations
        ]
    measurements = []
    for name, config in configurations:
        selected_jit = wy_backward if name.startswith("hoisted-") else jit

        def launch(config=config, selected_jit=selected_jit):
            return selected_jit[grid](
                **kwargs,
                IS_VARLEN=True,
                **config.kwargs,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )

        try:
            compiled = launch()
            torch.cuda.synchronize()
            errors = {}
            for output in output_names:
                actual, reference = kwargs[output].float(), expected[output].float()
                if not torch.isfinite(actual).all():
                    raise AssertionError(f"Non-finite {output}")
                errors[output] = {
                    "relative_max": (
                        (actual - reference).abs().max() / reference.abs().max().clamp_min(1e-8)
                    ).item(),
                    "relative_rms": (
                        (actual - reference).square().mean().sqrt()
                        / reference.square().mean().sqrt().clamp_min(1e-8)
                    ).item(),
                }
            valid = all(
                e["relative_max"] < 0.02 and e["relative_rms"] < 0.01 for e in errors.values()
            )
            samples = [triton.testing.do_bench_cudagraph(launch, rep=100) for _ in range(3)]
            row = {
                "name": name,
                "BK": config.kwargs["BK"],
                "BV": config.kwargs["BV"],
                "warps": config.num_warps,
                "stages": config.num_stages,
                "ms": samples,
                "valid": valid,
                "errors": errors,
                "registers": compiled.n_regs,
                "spills": compiled.n_spills,
                "shared_bytes": compiled.metadata.shared,
            }
        except Exception as error:  # noqa: BLE001 - retain failed launch layouts in the audit
            row = {"name": name, "valid": False, "error": str(error)}
        measurements.append(row)
        print(json.dumps(row), flush=True)
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": SHAPE,
        "grid": grid,
        "sources": sources,
        "measurements": measurements,
    }
