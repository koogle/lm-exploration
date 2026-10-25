"""Training-step throughput and MFU of a hybrid run config on synthetic packed documents.

    torchrun --nproc_per_node=<gpus> -m model_hybrid.benchmark --config <run.json> \\
        --steps 20 [--microbatch 4] [--document-tokens 1000] [--profile <out.json>]

Runs the real training step (trainer, optimizer, compile) without checkpoints or
evaluation. Document lengths are geometric with the given mean, close to the
recipe's mixture (ClimbMix ~600, code ~1,700, math ~1,400 tokens). Prints one JSON
line per step from the trainer; summarize with `summarize`. With `--profile`, rank 0
records PROFILE_STEPS steps after PROFILE_SKIP with the PyTorch profiler and writes
GPU time per kernel category and the slowest kernels.
"""

import argparse
import collections
import dataclasses
import json
import os
import re
import tempfile
import time

import numpy as np

from . import tokenizer as glm
from .train import RunConfig, train

PROFILE_SKIP, PROFILE_STEPS = 6, 3
# First match wins; names are CUDA kernel names as the profiler reports them.
KERNEL_CATEGORIES = (
    ("communication", r"nccl"),
    ("flex_attention", r"flex_attention|triton_tem_fused"),
    (
        "kda_fla",
        r"chunk_kda|kda|chunk_gated_delta|chunk_fwd|chunk_bwd|chunk_local|solve_tril|wy_|l2norm",
    ),
    ("loss_softmax_gradient", r"softmax_gradient"),
    ("matmul", r"gemm|cutlass|nvjet|sm90_xmma|cublas|matmul"),
    ("optimizer", r"multi_tensor|foreach|adam|newton|muon"),
    ("compiled_elementwise", r"triton_(poi|red|per)"),
    ("memory", r"memcpy|memset|Memcpy|Memset|copy_|fill"),
)


class SyntheticMixture:
    def __init__(self, context, vocab, document_tokens, seed=0):
        self.context, self.vocab, self.mean = context, vocab, document_tokens
        self.rng = np.random.default_rng(seed)

    def next_batch(self, rows):
        tokens = self.rng.integers(0, glm.TOKENIZER_SIZE - 64, (rows, self.context + 1))
        ends = self.rng.random((rows, self.context + 1)) < 1 / self.mean
        tokens[ends] = glm.EOS_ID
        return tokens[:, :-1], tokens[:, 1:], ["synthetic"] * rows

    def state_dict(self):
        return {}


def summarize(lines, skip=5):
    """Median throughput and MFU of the steps after the first `skip` (compile, warmup)."""
    steps = [json.loads(line) for line in lines if line.startswith('{"step"')]
    timed = [s for s in steps if s["step"] > skip]
    return {
        "steps_timed": len(timed),
        "tokens_per_second": float(np.median([s["tokens_per_second"] for s in timed])),
        "mfu": float(np.median([s["mfu"] for s in timed])),
        "loss_first": steps[0]["loss"] if steps else None,
        "loss_last": steps[-1]["loss"] if steps else None,
    }


def kernel_category(name):
    for category, pattern in KERNEL_CATEGORIES:
        if re.search(pattern, name):
            return category
    return "other"


def profile_summary(profiler, steps):
    """GPU kernel time per category and the 25 slowest kernels, per profiled step."""
    kernels = collections.Counter()
    for event in profiler.events():
        # User annotations (ProfilerStep#, module and optimizer ranges) span kernels.
        if event.is_user_annotation or event.device_type.name != "CUDA":
            continue
        if event.self_device_time_total > 0:
            kernels[event.name] += event.self_device_time_total / 1000 / steps
    categories = collections.Counter()
    for name, ms in kernels.items():
        categories[kernel_category(name)] += ms
    total = sum(categories.values())
    return {
        "steps": steps,
        "gpu_kernel_ms_per_step": total,
        "categories_ms": dict(categories.most_common()),
        "categories_share": {k: v / total for k, v in categories.most_common()},
        "slowest_kernels_ms": [
            {"name": name[:160], "ms": ms, "category": kernel_category(name)}
            for name, ms in kernels.most_common(25)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--global-rows", type=int)
    parser.add_argument("--document-tokens", type=int, default=1000)
    parser.add_argument("--profile", help="write a kernel-time summary JSON here (rank 0)")
    args = parser.parse_args()
    config = RunConfig.load(args.config)
    overrides = {"recipe": "", "hf_repo": "", "log_every_steps": 1, "checkpoint_seconds": 10**9}
    if args.microbatch:
        overrides["microbatch"] = args.microbatch
    if args.global_rows:
        overrides["global_rows"] = args.global_rows
    config = dataclasses.replace(config, **overrides)
    context = config.model["block_size"]
    on_step, profiler = None, None
    if args.profile and int(os.environ.get("RANK", "0")) == 0:
        import torch

        def write(finished):
            summary = profile_summary(finished, PROFILE_STEPS)
            with open(args.profile, "w") as out:
                json.dump(summary, out, indent=2)

        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        schedule = torch.profiler.schedule(wait=PROFILE_SKIP - 1, warmup=1, active=PROFILE_STEPS)
        profiler = torch.profiler.profile(
            activities=activities, schedule=schedule, on_trace_ready=write
        )
        profiler.start()

        def on_step(progress):
            profiler.step()

    with tempfile.TemporaryDirectory() as run_dir:
        train(
            config,
            run_dir,
            time.time() + 3600,
            max_tokens=args.steps * config.global_rows * context,
            data_source=lambda recipe: SyntheticMixture(
                context, config.model["vocab_size"], args.document_tokens
            ),
            save_final=False,
            on_step=on_step,
        )
    if profiler:
        profiler.stop()


if __name__ == "__main__":
    main()
