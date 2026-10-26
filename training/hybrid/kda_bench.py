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


def time_convolutions(dtype=torch.bfloat16, mean_document=1000):
    """The three short convolutions of one layer, compiled as in training."""
    generator = torch.Generator().manual_seed(1)
    s = SHAPE
    documents = packed_documents(s["batch"], s["length"], mean_document, generator)
    positions = document_positions(documents)
    channels = s["heads"] * s["width"]
    xs = [
        torch.randn(s["batch"], s["length"], channels, generator=generator)
        .cuda()
        .to(dtype)
        .requires_grad_()
        for _ in range(3)
    ]
    weights = [torch.randn(channels, 1, 4).cuda().requires_grad_() for _ in range(3)]
    convolution = torch.compile(kda.short_convolution)

    def forward_backward():
        outs = [convolution(x, w, positions) for x, w in zip(xs, weights, strict=True)]
        torch.autograd.grad(outs, [*xs, *weights], [torch.ones_like(o) for o in outs])

    return kernel_times(forward_backward)


def accuracy(implementation, dtype, length=512, tolerance=5e-2):
    """Relative max error of outputs and all input gradients against kda_reference."""
    generator = torch.Generator().manual_seed(2)
    documents = packed_documents(2, length, 150, generator)
    tensors, documents = layer_inputs(2, length, 4, 64, documents, dtype, generator)
    upstream = torch.randn(2, length, 4, 64, generator=generator).cuda()
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
    implementations = implementations or {"fla": kda.kda_kernel}
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
        "short_convolutions": time_convolutions(),
    }
