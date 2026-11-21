"""Bounded FP8 output/gradient check on production linear shapes before timing."""

import argparse
import copy
import json

import torch
from model_hybrid.float8_training import configure


def checks(recipe="rowwise"):
    torch.manual_seed(1234)
    results = []
    for inputs, outputs in ((1536, 4096), (4096, 1536), (1536, 512)):
        base = torch.nn.Sequential(torch.nn.Linear(inputs, outputs, bias=False)).cuda()
        candidate = copy.deepcopy(base)
        configure(candidate, recipe)
        x = torch.randn(2048, inputs, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn(2048, outputs, device="cuda", dtype=torch.bfloat16)

        def evaluate(model, x=x, dy=dy):
            leaf = x.detach().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(leaf)
            grads = torch.autograd.grad(y, (leaf, model[0].weight), dy)
            return (y, *grads)

        expected = evaluate(base)
        actual = evaluate(candidate)
        errors = [
            float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
            for a, b in zip(actual, expected, strict=True)
        ]
        finite = all(bool(torch.isfinite(t).all()) for t in actual)
        row = {
            "shape": [2048, inputs, outputs],
            "relative_l2_output_dx_dw": errors,
            "finite": finite,
            "threshold": 0.10,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        assert finite and max(errors) < 0.10, row
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", default="rowwise")
    checks(parser.parse_args().recipe)
