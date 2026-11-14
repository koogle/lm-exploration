"""Compiled production-shape linear comparison with backward-only kernel profiles.

Run through modal_checks::main --only fp8_backward. Diagnostic profiles are kept
separate from unprofiled CUDA event timings. All recipes share weights and inputs.
"""

import collections
import copy
import gc
import statistics

import torch

from .float8_training import RECIPES, configure
from .profile_categories import kernel_category

SHAPES = ((1536, 4096), (4096, 1536), (1536, 1536), (1536, 512))


def backward_profile(model, x, dy):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(x)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        torch.autograd.grad(y, (x, model._orig_mod[0].weight), dy)
        torch.cuda.synchronize()
    names = collections.Counter()
    for event in profiler.events():
        if event.device_type.name == "CUDA" and not event.is_user_annotation:
            names[event.name] += event.self_device_time_total / 1000
    groups = collections.Counter()
    for name, ms in names.items():
        groups[kernel_category(name)] += ms
    return {
        "kernel_sums_ms": dict(names.most_common()),
        "categories_ms": dict(groups),
        "scope": "One instrumented backward only; sums may overlap. Not a throughput measurement.",
    }


def measure(model, x, dy):
    for _ in range(4):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(x)
        grads = torch.autograd.grad(y, (x, model._orig_mod[0].weight), dy)
    torch.cuda.synchronize()
    timings = []
    for _ in range(20):
        start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(x)
        middle.record()
        grads = torch.autograd.grad(y, (x, model._orig_mod[0].weight), dy)
        end.record()
        end.synchronize()
        timings.append(
            {
                "forward_ms": start.elapsed_time(middle),
                "backward_ms": middle.elapsed_time(end),
                "total_ms": start.elapsed_time(end),
            }
        )
    return {key: statistics.median(r[key] for r in timings) for key in timings[0]}, (y, *grads)


def run():
    import torchao

    torch.manual_seed(1234)
    rows = []
    for tokens in (24576, 8192):
        for inputs, outputs in SHAPES:
            base = torch.nn.Sequential(torch.nn.Linear(inputs, outputs, bias=False)).cuda()
            x = torch.randn(tokens, inputs, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            dy = torch.randn(tokens, outputs, device="cuda", dtype=torch.bfloat16)
            expected = None
            for recipe in ("bf16", *RECIPES):
                torch._dynamo.reset()
                model = copy.deepcopy(base)
                if recipe != "bf16":
                    configure(model, recipe)
                compiled = torch.compile(model, fullgraph=True)
                timing, actual = measure(compiled, x, dy)
                if expected is None:
                    expected = tuple(t.detach().clone() for t in actual)
                errors = [
                    float(
                        (a.detach().float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
                    )
                    for a, b in zip(actual, expected, strict=True)
                ]
                finite = all(bool(torch.isfinite(t).all()) for t in actual)
                row = {
                    "shape": [tokens, inputs, outputs],
                    "recipe": recipe,
                    **timing,
                    "relative_l2_output_dx_dw": errors,
                    "finite": finite,
                    "backward_profile": backward_profile(compiled, x, dy),
                }
                rows.append(row)
                print({k: v for k, v in row.items() if k != "backward_profile"}, flush=True)
                assert finite and max(errors) < 0.10, row
                del actual, compiled, model
                gc.collect()
                torch.cuda.empty_cache()
            del base, x, dy, expected
    return {
        "hardware": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torchao": torchao.__version__,
        "rows": rows,
        "scope": "Isolated linears with FP32 parameters and BF16 autocast; Gaussian inputs, no optimizer or DDP. Shapes match microbatch six and tail two; not end-to-end MFU or convergence.",
    }
