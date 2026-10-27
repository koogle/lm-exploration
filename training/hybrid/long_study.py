"""Matched 100M-class follow-up to the PR77 mixer pilot."""

import json
import time
from pathlib import Path

from .modal_study import app, image, runs

CONFIG = {
    "vocab_size": 32768,
    "eos_id": 2,
    "block_size": 1024,
    "n_layer": 12,
    "n_embd": 768,
    "n_head": 12,
    "n_kv_head": 4,
    "n_kda_head": 12,
    "head_dim": 64,
    "mlp_hidden": 1792,
    "loss_chunk_size": 1024,
}
VARIANTS = ["baseline", "none", "rotating_triton", "rope", "middle_rope"]
STEPS = 12288
BATCH = 8
TOKENS_PER_STEP = BATCH * CONFIG["block_size"]
EVAL_BATCHES = 32


@app.function(
    image=image, gpu="H100", cpu=8, memory=32768, timeout=900, retries=0, volumes={"/runs": runs}
)
def rope_kernel_study(run, deadline):
    import numpy as np
    import torch
    from model_hybrid.modal_study import benchmark
    from model_hybrid.rope_kernel_study import compare_training, qualify

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert time.time() < deadline - 120
    result = {"correctness": qualify()}
    runs.reload()
    folder = Path("/runs") / run
    (folder / "rope-kernel-checks.json").write_text(json.dumps(result))
    runs.commit()
    print("RoPE kernel numerical checks passed", flush=True)
    result["training"] = compare_training(
        np.load(folder / "train.npy", mmap_mode="r"), CONFIG, deadline
    )
    (folder / "rope-kernel-checks.json").write_text(json.dumps(result))
    runs.commit()
    assert result["training"]["max_ce_difference"] < 0.01, result["training"]["max_ce_difference"]
    torch.cuda.empty_cache()
    result["benchmark"] = benchmark.local(run, deadline, CONFIG, ["none", "rope", "rope_triton"])
    (folder / "rope-kernel.json").write_text(json.dumps(result))
    runs.commit()
    return result


@app.function(image=image, gpu="H100", cpu=2, memory=8192, timeout=300, retries=0)
def inspect_inputs(deadline):
    """Record actual recurrence input dtypes under the study's CUDA autocast."""
    import model_hybrid.study_mixers as mixers
    import torch
    from model_hybrid.model import DocumentMask, HybridConfig, HybridLM, rope_tables

    assert time.time() < deadline - 60
    torch.set_num_threads(2)
    records = {}
    model = HybridLM(HybridConfig(**CONFIG))
    x = torch.randn(2, 16, 768, device="cuda")
    mask = DocumentMask(torch.zeros(2, 16, dtype=torch.long, device="cuda"))
    tables = rope_tables(16, 64, 1e6, "cuda")
    for variant in ["baseline", "baseline_bf16", "none", "rope"]:
        import copy

        selected = mixers.configure(copy.deepcopy(model), variant).blocks[0].mixer.cuda()

        def capture(q, k, v, *args, variant=variant):
            records[variant] = {name: str(t.dtype) for name, t in zip(["q", "k", "v"], [q, k, v])}
            return torch.zeros_like(v)

        original = mixers.kda_kernel
        mixers.kda_kernel = capture
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                selected(x, tables, mask)
        finally:
            mixers.kda_kernel = original
        del selected
    return {"input_dtypes": records, "torch": str(torch.__version__)}


@app.function(image=image, cpu=2, memory=4096, timeout=1200, retries=0, volumes={"/runs": runs})
def copy_prepared(source, run, deadline):
    """Copy an explicitly selected, hash-verified corpus after a launcher failure."""
    import hashlib
    import shutil

    import numpy as np

    assert time.time() < deadline - 300
    runs.reload()
    origin, target = Path("/runs") / source, Path("/runs") / run
    info = json.loads((origin / "data.json").read_text())
    target.mkdir(parents=True, exist_ok=True)
    for split, minimum in [
        ("train", STEPS * TOKENS_PER_STEP),
        ("validation", EVAL_BATCHES * TOKENS_PER_STEP),
    ]:
        arr = np.load(origin / f"{split}.npy", mmap_mode="r")
        assert len(arr) > minimum
        assert hashlib.sha256(arr.tobytes()).hexdigest() == info[split]["tokens_sha256"]
        shutil.copyfile(origin / f"{split}.npy", target / f"{split}.npy")
    assert not set(info["train"]["document_hashes"]) & set(info["validation"]["document_hashes"])
    shutil.copyfile(origin / "data.json", target / "data.json")
    runs.commit()
    return info


@app.function(
    image=image, gpu="H100", cpu=8, memory=32768, timeout=1800, retries=0, volumes={"/runs": runs}
)
def qualify_long(run, deadline):
    import torch
    from model_hybrid.kernel_study import qualify
    from model_hybrid.model import HybridConfig, HybridLM
    from model_hybrid.study_mixers import RotatingMemory, configure

    assert time.time() < deadline - 120
    torch.set_num_threads(4)
    checks = qualify()
    torch.manual_seed(421)
    ref = RotatingMemory(768).cuda()
    custom = RotatingMemory(768).cuda()
    custom.load_state_dict(ref.state_dict())
    custom.use_kernel = True
    u = torch.randn(8, 1024, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = u.detach().clone().requires_grad_()
    positions = torch.arange(1024, device="cuda").repeat(8, 1) % 257
    y, z = ref(u, positions), custom(v, positions)
    grad = torch.randn_like(y)
    a = torch.autograd.grad(y, (u, *ref.parameters()), grad)
    b = torch.autograd.grad(z, (v, *custom.parameters()), grad)
    shape_errors = {}
    for name, x, w in zip(["output", "input", "log_tau", "theta", "mix"], [y, *a], [z, *b]):
        absolute = (x.float() - w.float()).abs().max().item()
        relative = ((x.float() - w.float()).norm() / x.float().norm().clamp_min(1e-8)).item()
        assert relative < 0.0002 or absolute < 2e-5, (name, relative, absolute)
        shape_errors[name] = {"relative_l2": relative, "max_abs": absolute}
    del ref, custom, u, v, y, z, a, b, grad
    counts = {}
    for variant in VARIANTS:
        model = configure(HybridLM(HybridConfig(**CONFIG)), variant)
        counts[variant] = sum(p.numel() for p in model.parameters())
        del model
    result = {
        "kernel_checks": checks,
        "shape_errors": shape_errors,
        "parameters": counts,
        "config": CONFIG,
    }
    folder = Path("/runs") / run
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "qualification.json").write_text(json.dumps(result))
    runs.commit()
    return result


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=10800,
    retries=0,
    max_containers=10,
    volumes={"/runs": runs},
)
def train_long(run, variant, seed, deadline):
    return train_impl(run, variant, seed, deadline)


def train_impl(run, variant, seed, deadline, *, source=None, steps=STEPS, runtime=10200):
    import math
    from dataclasses import asdict

    import numpy as np
    import torch
    from model_hybrid.model import HybridConfig, HybridLM
    from model_hybrid.study_mixers import configure

    assert variant in VARIANTS or variant in {"baseline_bf16", "rope_triton"}
    assert time.time() < deadline - 300
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    runs.reload()
    root = Path("/runs") / run
    folder = root / f"{variant}-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {s: np.load(root / f"{s}.npy", mmap_mode="r") for s in ["train", "validation"]}
    start_step = STEPS if source else 0
    assert len(arrays["train"]) > (steps - start_step) * TOKENS_PER_STEP
    assert len(arrays["validation"]) > EVAL_BATCHES * TOKENS_PER_STEP
    cfg = HybridConfig(**CONFIG)
    model = configure(HybridLM(cfg), variant).cuda()
    opt = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
    )

    if source:
        original_variant = "rope" if variant == "rope_triton" else variant
        origin = Path("/runs") / source / f"{original_variant}-{seed}"
        checkpoint = torch.load(origin / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert checkpoint["step"] == start_step
        assert checkpoint["config"] == asdict(cfg)
        assert checkpoint["variant"] == original_variant and checkpoint["seed"] == seed
        model.load_state_dict(checkpoint["model"], strict=True)
        opt.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        del checkpoint

    def batch(split, index):
        start = index * TOKENS_PER_STEP
        arr = arrays[split]
        assert start + TOKENS_PER_STEP + 1 <= len(arr)
        rows = np.stack([arr[start + i * 1024 : start + (i + 1) * 1024 + 1] for i in range(BATCH)])
        x = torch.tensor(rows[:, :-1], device="cuda", dtype=torch.long)
        y = torch.tensor(rows[:, 1:], device="cuda", dtype=torch.long)
        y[y == 1] = -100
        return x, y

    def evaluate():
        model.eval()
        ce = count = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(EVAL_BATCHES):
                x, y = batch("validation", i)
                out = model(x, y)
                ce += out["ce"].item()
                count += out["count"].item()
        model.train()
        return ce / count

    result = {
        "variant": variant,
        "seed": seed,
        "config": asdict(cfg),
        "steps_planned": steps,
        "token_positions_planned": steps * TOKENS_PER_STEP,
        "source_run": source,
        "start_step": start_step,
        "parameters": sum(p.numel() for p in model.parameters()),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "initial_ce": evaluate(),
        "curve": [],
        "status": "running",
    }
    if source:
        previous = json.loads((origin / "metrics.json").read_text())["curve"][-1]["validation_ce"]
        assert abs(result["initial_ce"] - previous) < 0.002, (result["initial_ce"], previous)
    started = time.time()
    elapsed = train_ce = train_count = timed_targets = 0.0
    interval_steps = 0
    torch.cuda.reset_peak_memory_stats()

    def save(step, checkpoint=False):
        result["steps_completed"] = step
        result["wall_seconds"] = time.time() - started
        result["max_memory_bytes"] = torch.cuda.max_memory_allocated()
        (folder / "metrics.json").write_text(json.dumps(result, indent=2))
        if checkpoint:
            temp = folder / "checkpoint.tmp"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step,
                    "config": asdict(cfg),
                    "variant": variant,
                    "seed": seed,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                temp,
            )
            temp.replace(folder / "checkpoint.pt")
        runs.commit()

    save(start_step)
    for step in range(start_step, steps):
        if time.time() > deadline - 180 or time.time() - started > runtime:
            result["status"] = "deadline"
            save(step, checkpoint=True)
            return result
        lr = (
            3e-4
            * min(1.0, (step + 1) / 400)
            * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
        )
        if source:
            local_step = step - start_step
            warm = min(1.0, (local_step + 1) / 400)
            progress = max(0.0, (local_step - 400) / (steps - start_step - 400))
            lr = (3e-5 + (3e-4 - 3e-5) * warm) * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            )
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = batch("train", step - start_step)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, y)
            loss = out["loss"] / out["count"]
        if not torch.isfinite(loss):
            result["status"] = "nonfinite_loss"
            save(step, checkpoint=True)
            raise RuntimeError("nonfinite loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        count = out["count"].item()
        train_ce += out["ce"].item()
        train_count += count
        interval_steps += 1
        if step >= start_step + 10:
            elapsed += seconds
            timed_targets += count
        if (step + 1) % (2048 if source else 512) == 0 or step == start_step or step + 1 == steps:
            row = {
                "step": step + 1,
                "token_positions": (step + 1) * TOKENS_PER_STEP,
                "train_ce": train_ce / train_count,
                "train_interval_steps": interval_steps,
                "validation_ce": evaluate(),
                "lr": lr,
                "grad_norm": grad_norm.item(),
                "tokens_per_second": timed_targets / elapsed if elapsed else None,
                "optimizer_seconds": elapsed,
                "wall_seconds": time.time() - started,
            }
            result["curve"].append(row)
            result["status"] = "complete" if step + 1 == steps else "running"
            print(variant, seed, row, flush=True)
            save(
                step + 1,
                checkpoint=(step + 1) % 2048 == 0 or step == start_step or step + 1 == steps,
            )
            train_ce = train_count = 0.0
            interval_steps = 0
    return result
