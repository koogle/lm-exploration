"""Production vocabulary-loss shapes: complete loss+gradient cost and numerical drift."""

import statistics

import torch

from . import fused_loss
from .kernels.fp8_cast import quantize


def run():
    torch.manual_seed(1234)
    results = []
    # Include an exact-zero cast to cover scale protection and FP8 finite zeros.
    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        q, scale = quantize(torch.zeros(32, 32, device="cuda"), dtype)
        assert torch.isfinite(q.float()).all() and (q.float() == 0).all() and scale > 0
    for rows in (8192, 24576):
        for distribution in ("normal", "peaked", "outlier"):
            h = torch.randn(rows, 1536, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(154880, 1536, device="cuda") / 1536**0.5
            if distribution == "peaked":
                h.mul_(8)
            if distribution == "outlier":
                h[:, 0].mul_(100)
                w[::257, 0].mul_(30)
            labels = torch.randint(154880, (rows,), device="cuda")
            labels[::11] = -100
            buffer = torch.zeros_like(w)
            expected = None
            for mode in ("bf16", "logits", "logits_dx", "bf16_repeat"):
                fused_loss.FP8_PROJECTION = "" if mode.startswith("bf16") else mode
                timings = []
                for i in range(10):
                    buffer.zero_()
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record()
                    actual = fused_loss._loss_grad(h, w, labels, 1e-4, 2048, buffer)
                    end.record()
                    end.synchronize()
                    if i >= 3:
                        timings.append(start.elapsed_time(end))
                actual = (*actual, buffer)
                if expected is None:
                    expected = tuple(x.clone() for x in actual)
                errors = [
                    float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
                    for a, b in zip(actual, expected, strict=True)
                ]
                row = {
                    "rows": rows,
                    "distribution": distribution,
                    "mode": mode,
                    "total_loss_and_grad_ms": statistics.median(timings),
                    "relative_l2_dh_ce_z_dw": errors,
                    "finite": all(bool(torch.isfinite(x).all()) for x in actual),
                    "ce_per_valid_token": float(actual[1] / (labels != -100).sum()),
                }
                results.append(row)
                print(row, flush=True)
            del expected, actual, buffer, h, w, labels
            torch.cuda.empty_cache()
    fused_loss.FP8_PROJECTION = ""
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rows": results,
        "scope": "Same production loss loop; includes FP8 weight quantization/layout once per head, activation/gradient casts per chunk, all loss gradients. Eager isolated operation, not compiled full-model throughput or learning-quality proof.",
    }
