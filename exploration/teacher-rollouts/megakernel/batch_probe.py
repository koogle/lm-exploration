"""Prototype: independent fused backbones with a shared batched vocabulary head.

Backbones still launch sequentially. This is partial batching, not a batched
transformer kernel, and is a diagnostic rather than a production backend.
"""

import time

import torch

from core import atomic_json
from megakernel.buffer_probe import metrics
from megakernel.model import Decoder


@torch.inference_mode()
def step(lanes, tokens, batched_head):
    outputs = []
    for lane, token in zip(lanes, tokens, strict=True):
        if not batched_head:
            outputs.append(lane.step_device(token))
            continue
        if lane.position >= lane.max_seq_len:
            raise ValueError("Cache overflow")
        code = lane.library.decode_device(
            lane.pointers,
            token.data_ptr(),
            lane.position,
            lane.max_seq_len,
            0,
            torch.cuda.current_stream().cuda_stream,
        )
        if code:
            raise RuntimeError(lane.library.error_string(code).decode())
        lane.position += 1
        outputs.append(lane.normalized.to(torch.bfloat16))
    if batched_head:
        return torch.nn.functional.linear(torch.stack(outputs), lanes[0].head).float()
    return torch.stack(outputs)


@torch.inference_mode()
def qualify(model, head, tokenizer, path):
    lanes = [
        Decoder(model, head, "/experiment/megakernel/kernel.so", 256) for _ in range(32)
    ]
    prompts = [tokenizer.encode(f"Python factorial input {i}: ") for i in range(32)]
    tokens = torch.tensor(
        [tokenizer.encode(f"def solve(n): return n * {i + 1}")[:8] for i in range(32)],
        device="cuda",
        dtype=torch.int64,
    ).contiguous()
    # Explicitly equal length teacher-forced continuations, with distinct lane data.
    if tokens.shape != (32, 8):
        raise ValueError("Unexpected fixture token lengths")
    report = {
        "scope": "Sequential fused backbones; batched vocabulary head only",
        "rows": [],
    }
    for size in [1, 4, 8, 16, 32]:
        active = lanes[:size]
        values, timing = {}, {"separate": [], "batched_head": []}
        for repeat in range(2):
            modes = [False, True] if repeat == 0 else [True, False]
            for batched in modes:
                for lane, ids in zip(active, prompts[:size], strict=True):
                    lane.reset()
                    for token in ids:
                        lane.step(token)
                torch.cuda.synchronize()
                before = time.perf_counter()
                rows = [
                    step(active, tokens[:size, t].contiguous(), batched)
                    for t in range(8)
                ]
                torch.cuda.synchronize()
                name = "batched_head" if batched else "separate"
                timing[name].append(time.perf_counter() - before)
                values[name] = torch.stack(rows).reshape(-1, head.shape[0])
        error = metrics(values["batched_head"], values["separate"])
        row = {
            "batch_size": size,
            "error": error,
            "seconds": timing,
            "speedup": sum(timing["separate"]) / sum(timing["batched_head"]),
            "passed": error["finite"]
            and error["nrmse"] <= 0.005
            and error["top1_fraction"] >= 0.99,
        }
        report["rows"].append(row)
        atomic_json(path, report)
        print("BATCH_HEAD_PROBE", row, flush=True)
    report["passed"] = all(r["passed"] for r in report["rows"])
    atomic_json(path, report)
    del lanes, values, rows
    torch.cuda.empty_cache()
    return report
