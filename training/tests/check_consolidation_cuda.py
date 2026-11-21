"""H100 correctness gates and isolated timings for the three consolidation candidates."""

import copy
import json
import time
from functools import partial

import torch
import triton.testing
from model_10m.model import RMSNorm
from model_hybrid import kda_intra, kda_wy, model
from model_hybrid.consolidation import configure
from model_hybrid.fused_loss import fused_cross_entropy, joint_cross_entropy
from model_hybrid.kernels.rmsnorm import rms_norm


def errors(actual, reference, tolerance):
    values = []
    for a, b in zip(actual, reference, strict=True):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        error = ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-8)).item()
        if error > tolerance:
            raise AssertionError(f"normalized error {error} exceeds {tolerance}")
        values.append(error)
    return values


def timing(function):
    function()
    torch.cuda.synchronize()
    # The isolated benchmark captures on Triton's stream after default-stream
    # correctness checks. Redirect stale leaf stream references for capture only.
    torch.autograd.graph.set_override_stale_capture_stream(True)
    try:
        return [triton.testing.do_bench_cudagraph(function, rep=100) for _ in range(3)]
    finally:
        torch.autograd.graph.set_override_stale_capture_stream(False)


def norm_check():
    results = []
    for rows, width in ((13, 128), (257, 1536), (24576, 1536)):
        torch.manual_seed(rows)
        x = torch.randn(rows, width, device="cuda", requires_grad=True)
        norm = RMSNorm(width).cuda()
        norm.weight.data.uniform_(0.5, 1.5)
        dy = torch.randn_like(x)
        original = torch.compile(norm)
        revised = torch.compile(lambda value, norm=norm: rms_norm(value, norm.weight, norm.eps))

        def run(function, x=x, norm=norm, dy=dy):
            y = function(x)
            return (y, *torch.autograd.grad(y, (x, norm.weight), dy))

        reference = run(original)
        actual = run(revised)
        row = {"shape": [rows, width], "errors": errors(actual, reference, 2e-5)}
        if rows == 24576:
            row["original_ms"] = timing(partial(run, original))
            row["revised_ms"] = timing(partial(run, revised))
        results.append(row)
    return results


def joint_check(training_shape=False):
    torch.manual_seed(41)
    rows, future_rows, chunk = (24576, 24576, 2048) if training_shape else (2048, 513, 512)
    first = torch.randn(rows, 1536, device="cuda", requires_grad=True)
    second = torch.randn(future_rows, 1536, device="cuda", requires_grad=True)
    weight = (torch.randn(154880, 1536, device="cuda") * 0.02).requires_grad_()
    y = torch.randint(0, 154880, (rows,), device="cuda")
    future = torch.randint(0, 154880, (future_rows,), device="cuda")
    y[::7], future[::3] = -100, -100
    scales = (torch.tensor(1 / 1048576, device="cuda"), torch.tensor(0.3 / 524288, device="cuda"))

    def separate():
        a = fused_cross_entropy(first, weight, y, 1e-4, chunk)
        b = fused_cross_entropy(second, weight, future, 1e-4, chunk)
        loss = 8 * (scales[0] * a[0] + scales[1] * b[0])
        return loss, *torch.autograd.grad(loss, (first, second, weight))

    def joint_objective():
        return 8 * joint_cross_entropy(first, second, weight, y, future, scales, 1e-4, chunk)[0]

    objective = torch.compile(joint_objective) if training_shape else joint_objective

    def joint():
        loss = objective()
        return loss, *torch.autograd.grad(loss, (first, second, weight))

    reference = separate()
    torch.cuda.synchronize()
    start = time.perf_counter()
    actual = joint()
    torch.cuda.synchronize()
    result = {
        "errors": errors(actual, reference, 2e-5),
        "rows": [rows, future_rows],
        "chunk": chunk,
        "first_candidate_call_seconds": time.perf_counter() - start,
        "compiled": training_shape,
    }
    del reference, actual
    if training_shape:
        result["compiled_candidate_ms"] = timing(joint)
    else:
        result["original_ms"], result["revised_ms"] = timing(separate), timing(joint)
    return result


def model_check(candidates=("joint-loss", "projections", "norm", "all")):
    kda_wy.install()
    kda_intra.install()
    torch.manual_seed(37)
    config = model.HybridConfig(
        vocab_size=521,
        eos_id=2,
        block_size=512,
        n_layer=2,
        n_embd=256,
        n_head=2,
        n_kv_head=1,
        head_dim=128,
        n_kda_head=2,
        mlp_hidden=512,
        layer_pattern=("kda", "full"),
        mtp=True,
    )
    original = model.HybridLM(config).cuda()
    # Exercise the training attention path rather than FlexAttention's short
    # decode specialization, whose automatic tile choice rejects length 65.
    tokens = torch.randint(3, 521, (2, 257), device="cuda")
    tokens[:, 127] = 2
    labels = torch.randint(3, 521, tokens.shape, device="cuda")
    labels[:, -3:] = -100
    future = labels.roll(-1, 1)
    future[:, -1] = -100
    scales = (torch.tensor(1 / 256, device="cuda"), torch.tensor(0.3 / 254, device="cuda"))

    def run(lm, joint):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = lm(tokens, labels, future, loss_scales=scales if joint else None)
            loss = (
                result["objective"]
                if joint
                else scales[0] * result["loss"] + scales[1] * result["mtp_loss"]
            )
        return (loss, *torch.autograd.grad(loss, tuple(lm.parameters())))

    reference = run(torch.compile(original), False)
    results = {}
    for candidate in candidates:
        revised = copy.deepcopy(original)
        configure(revised, candidate)
        actual = run(torch.compile(revised), candidate in ("joint-loss", "all"))
        results[candidate] = errors(actual, reference, 0.03)
        print(json.dumps({"model_candidate": candidate, "errors": results[candidate]}), flush=True)
        del revised, actual
    return results


def checks(only=""):
    assert torch.cuda.get_device_capability() == (9, 0)
    results = {"gpu": torch.cuda.get_device_name()}
    if only == "joint-production":
        results["joint"] = joint_check(training_shape=True)
        print(json.dumps({"joint": results["joint"]}), flush=True)
        results["model"] = model_check(("joint-loss", "all"))
        return results
    for name, function in (("norm", norm_check), ("joint", joint_check), ("model", model_check)):
        if only and name != only:
            continue
        results[name] = function()
        print(json.dumps({name: results[name]}), flush=True)
    return results
