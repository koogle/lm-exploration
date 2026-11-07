"""Compare the production loss reduction with a vector-grouped reduction on H100."""

import statistics

import torch

from .cuda_extension import load


def run():
    torch.manual_seed(1234)
    extension = load("softmax_gradient")
    results = []
    for dtype in (torch.bfloat16, torch.float32):
        for vocab in (154880, 8193):
            for scale in (1.0, 20.0):
                rows = 2048 if vocab == 154880 else 13
                source = torch.randn(rows, vocab, device="cuda", dtype=dtype) * scale
                labels = torch.randint(vocab, (rows,), device="cuda")
                labels[::7] = -100
                outputs = []
                for grouped in (False, True):
                    x = source.clone()
                    stats = extension.softmax_gradient(x, labels, 1e-4, grouped)
                    outputs.append((x, *stats))
                errors = [
                    float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12))
                    for a, b in zip(*outputs, strict=True)
                ]
                assert all(torch.isfinite(x).all() for out in outputs for x in out)
                assert max(errors) < (0.002 if dtype == torch.bfloat16 else 1e-4), errors
                assert bool((outputs[1][0][labels == -100] == 0).all())
                # Time kernel only, excluding the same reset copy for both variants.
                timings = {}
                x = torch.empty_like(source)
                for grouped in (False, True, False):
                    elapsed = []
                    for i in range(24):
                        x.copy_(source)
                        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        extension.softmax_gradient(x, labels, 1e-4, grouped)
                        end.record()
                        end.synchronize()
                        if i >= 4:
                            elapsed.append(start.elapsed_time(end))
                    label = (
                        "grouped"
                        if grouped
                        else ("reference_repeat" if "reference" in timings else "reference")
                    )
                    timings[label] = statistics.median(elapsed)
                record = {
                    "dtype": str(dtype),
                    "shape": [rows, vocab],
                    "scale": scale,
                    "relative_l2_grad_lse_target": errors,
                    "median_ms": timings,
                }
                print(record, flush=True)
                results.append(record)
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": results}
