"""Correctness tests and a throughput check for the fixed cross-entropy CUDA kernel.

Run from training/:
    modal run -m modal_scripts.modal_stable_ce_check [--output <result.json>]

One H100, 15-minute timeout, no retries. The launcher records a deadline 24 hours
after launch, and the worker refuses to start after it. The image is a CUDA devel
image because the extension is JIT-built with nvcc. Results are structured JSON
in the ignored artifacts/.
"""

import json
import time
from pathlib import Path

import modal

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
SHAPES = ((16384, 32768), (8192, 50304))

app = modal.App("lm-exploration-stable-ce-check")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.6.0", "numpy", "ninja")
    .env({"PYTHONPATH": "/project", "TORCH_CUDA_ARCH_LIST": "9.0"})
    .add_local_dir("trainer", "/project/trainer")
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tests", "/project/tests")
)


def timed(fn, logits, targets, repeats=20):
    import torch

    def step():
        z = logits.detach().requires_grad_()
        fn(z, targets).backward()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        step()
    end.record()
    torch.cuda.synchronize()
    return {
        "ms_forward_backward": start.elapsed_time(end) / repeats,
        "peak_extra_mib": (torch.cuda.max_memory_allocated() - base) / 2**20,
    }


@app.function(image=image, gpu="H100", cpu=4, memory=32768, timeout=900, retries=0)
def check(deadline):
    import subprocess

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    tests = subprocess.run(
        [
            "python",
            "-m",
            "unittest",
            "-v",
            "tests.test_stable_ce_cuda",
            "tests.test_stable_softmax",
        ],
        check=False,
        cwd="/project",
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(tests.stderr)
    if tests.returncode:
        raise RuntimeError("Kernel tests failed")

    import torch
    from torch.nn import functional as F

    from trainer import stable_ce_cuda, stable_softmax

    losses = {
        # Autocast runs F.cross_entropy in fp32 on bf16 logits.
        "torch_cross_entropy": lambda z, t: F.cross_entropy(z.float(), t),
        "pytorch_port": stable_softmax.cross_entropy,
        "cuda_kernel": stable_ce_cuda.cross_entropy,
    }
    torch.manual_seed(0)
    benchmarks = []
    for rows, vocab in SHAPES:
        logits = torch.randn(rows, vocab, device="cuda", dtype=torch.bfloat16)
        targets = torch.randint(0, vocab, (rows,), device="cuda")
        for name, fn in losses.items():
            benchmarks.append(
                {"rows": rows, "vocab": vocab, "loss": name, **timed(fn, logits, targets)}
            )
            print(benchmarks[-1])
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "logits_dtype": "bfloat16",
        "tests": tests.stderr.strip().splitlines()[-1],
        "benchmarks": benchmarks,
    }


@app.local_entrypoint()
def main(output: str = ""):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    result = check.remote(deadline)
    path = Path(output or f"artifacts/results/stable-ce-check-{int(time.time())}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
