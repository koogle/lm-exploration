"""GPU validation of the fixed cross-entropy kernel in the shared training image.

Run from training/:
    modal run -m modal_scripts.modal_stable_ce_check [--output <result.json>]
    modal run -m modal_scripts.modal_stable_ce_check::speed [--steps 20]

`speed` times full optimizer steps of the 124M recipe on three H100s
(trainer/loss_benchmark.py), same 30-minute timeout and deadline.

One H100, 30-minute timeout, no retries. The launcher records a deadline 24 hours
after launch, and the worker refuses to start after it. The worker checks that the
image's prebuilt kernel is the one loaded, runs the kernel, port and model tests and
the trainer's CUDA gate (bf16, accumulation, chunked loss, compilation) against an
F.cross_entropy reference. It then times the loss alone and a full 124M training
microbatch with each loss. Results are structured JSON in the ignored artifacts/.
"""

import json
import time
from pathlib import Path

import modal

from distributed.image import image

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
LOSS_SHAPES = ((16384, 32768), (8192, 50304))

app = modal.App("lm-exploration-stable-ce-check")


def timed(step, repeats=20):
    import torch

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
        "ms": start.elapsed_time(end) / repeats,
        "peak_extra_mib": (torch.cuda.max_memory_allocated() - base) / 2**20,
    }


def run(command, timeout):
    import subprocess

    result = subprocess.run(
        command, check=False, cwd="/project", capture_output=True, text=True, timeout=timeout
    )
    print(result.stdout, result.stderr)
    if result.returncode:
        raise RuntimeError(f"Failed: {' '.join(command)}")
    return result


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=1800, retries=0)
def check(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import torch
    from model_100m.model import GPT, ModelConfig
    from torch.nn import functional as F
    from training_data.subword import EOS_TOKEN_ID, VOCAB_SIZE

    from trainer import stable_ce_cuda, stable_softmax
    from trainer.loss import cross_entropy

    prebuilt = stable_ce_cuda.extension().__file__
    if not prebuilt.startswith("/opt/stable_ce/"):
        raise RuntimeError(f"Expected the image's prebuilt kernel, loaded {prebuilt}")
    tests = run(
        [
            "python",
            "-m",
            "unittest",
            "-v",
            "tests.test_stable_ce_cuda",
            "tests.test_stable_softmax",
        ],
        900,
    )
    gate = run(["torchrun", "--nproc_per_node=1", "tests/check_optimization_cuda.py"], 900)

    def standard(z, t, **kwargs):
        # Autocast runs F.cross_entropy in fp32 on bf16 logits.
        return F.cross_entropy(z.float(), t, **kwargs)

    losses = {"torch_cross_entropy": standard, "cuda_kernel": cross_entropy}
    losses_alone = []
    torch.manual_seed(0)
    for rows, vocab in LOSS_SHAPES:
        logits = torch.randn(rows, vocab, device="cuda", dtype=torch.bfloat16)
        targets = torch.randint(0, vocab, (rows,), device="cuda")
        for name, fn in {**losses, "pytorch_port": stable_softmax.cross_entropy}.items():

            def step(fn=fn, logits=logits, targets=targets):
                fn(logits.detach().requires_grad_(), targets).backward()

            losses_alone.append({"rows": rows, "vocab": vocab, "loss": name, **timed(step)})

    # The active 124M recipe: 14 layers, 768 wide, context 2048, microbatch 16, bf16.
    model = GPT(
        ModelConfig(
            vocab_size=VOCAB_SIZE,
            block_size=2048,
            n_layer=14,
            n_head=12,
            n_embd=768,
            eos_id=EOS_TOKEN_ID,
        )
    ).cuda()
    x = torch.randint(0, VOCAB_SIZE, (16, 2048), device="cuda")
    y = torch.randint(0, VOCAB_SIZE, (16, 2048), device="cuda")
    microbatch = []
    for name, fn in losses.items():

        def step(fn=fn):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = fn(model(x).flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum")
            loss.backward()

        microbatch.append({"loss": name, **timed(step, repeats=10), "tokens": x.numel()})
        model.zero_grad(set_to_none=True)
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "prebuilt_kernel": prebuilt,
        "tests": tests.stderr.strip().splitlines()[-1],
        "optimization_gate": json.loads(gate.stdout.strip().splitlines()[-1]),
        "loss_alone_bf16": losses_alone,
        "train_microbatch_124m_bf16": microbatch,
    }


@app.function(image=image, gpu="H100:3", cpu=12, memory=65536, timeout=1800, retries=0)
def train_speed(deadline, steps):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    recipe = "/project/workflows/model_configs/modern-124m.json"
    command = ["torchrun", "--nproc_per_node=3", "-m", "trainer.loss_benchmark", recipe]
    result = run([*command, str(steps)], 1500)
    return json.loads(result.stdout.strip().splitlines()[-1])


@app.local_entrypoint()
def speed(steps: int = 20, output: str = ""):
    """Full 124M optimizer steps on three H100s, F.cross_entropy versus the kernel."""
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    result = train_speed.remote(deadline, steps)
    path = Path(output or f"artifacts/results/stable-ce-train-speed-{int(time.time())}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def main(output: str = ""):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    result = check.remote(deadline)
    path = Path(output or f"artifacts/results/stable-ce-check-{int(time.time())}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
