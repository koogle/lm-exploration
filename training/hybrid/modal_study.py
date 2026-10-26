"""Bounded PR77 mixer ablations; matched data and initialization."""

import json
import time
from pathlib import Path

import modal

app = modal.App("kda-mixer-study")
registry = modal.Dict.from_name("kda-mixer-study-deadlines", create_if_missing=True)
data = modal.Volume.from_name("lm-exploration-mixture-data")
runs = modal.Volume.from_name("lm-exploration-training")
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "torch==2.14.1",
        "numpy",
        "einops",
        "ninja",
        "fla-core==0.6.0",
        "packaging",
        "tokenizers==0.22.2",
        "huggingface-hub",
    )
    .env({"PYTHONPATH": "/project", "TORCH_CUDA_ARCH_LIST": "9.0"})
    .add_local_dir("10m", "/project/model_10m", ignore=["__pycache__"])
    .add_local_dir("hybrid", "/project/model_hybrid", ignore=["__pycache__"])
    .add_local_dir("data", "/project/training_data", ignore=["__pycache__", "raw", "cache"])
    .add_local_dir("tokenizer/climbmix-code-v1", "/project/tokenizer/climbmix-code-v1")
)
SNAP = "climbmix-code/climbmix-code-v6/snapshots/climbmix-124m-wsd20b-02"


@app.function(timeout=14400, retries=0)
def supervise(run, deadline):
    registry[run + "/ready"] = True
    while time.time() < deadline:
        if registry.get(run + "/done", False):
            return
        time.sleep(min(10, max(0, deadline - time.time())))
    for call in registry.get(run + "/calls", []):
        try:
            modal.FunctionCall.from_id(call).cancel()
        except Exception as error:  # noqa: BLE001
            print("Cancellation failed:", error, flush=True)
    registry[run + "/expired"] = True


@app.function(
    image=image,
    timeout=1200,
    retries=0,
    volumes={"/data": data, "/runs": runs},
    cpu=4,
    memory=16384,
)
def prepare(run, deadline):
    assert time.time() < deadline
    import hashlib
    import random
    import sqlite3
    import zlib

    import numpy as np
    from model_hybrid.test_study_mixers import checks
    from training_data.subword import load_frozen

    tests = checks()
    print(tests, flush=True)
    tokenizer = load_frozen()
    root = Path("/data") / SNAP
    manifest = json.loads((root / "manifest.json").read_text())
    db = sqlite3.connect(f"file:{root}/raw.sqlite?mode=ro&immutable=1", uri=True)
    sources = [r[0] for r in db.execute("SELECT DISTINCT source FROM documents")]
    print("sources", sources, flush=True)
    folder = Path("/runs") / run
    folder.mkdir(parents=True, exist_ok=True)
    info = {
        "tests": tests,
        "snapshot": SNAP,
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "training_ready": manifest.get("training_ready"),
        "sources": sources,
    }
    for split, total in [("train", 5000000), ("validation", 262144)]:
        documents = []
        per = max(1, total // len(sources))
        counts = {}
        for source in sources:
            count = 0
            for h, raw in db.execute(
                "SELECT hash,text FROM documents WHERE source=? AND split=? AND conflict=0 ORDER BY rowid",
                (source, split),
            ):
                ids = tokenizer.encode(zlib.decompress(raw).decode(), add_special_tokens=False).ids
                documents.append((h, [1] + ids + [2]))
                count += len(ids) + 2
                if count >= per:
                    break
            counts[source] = count
        assert documents, split
        random.Random(2026).shuffle(documents)
        tokens = np.array([t for _, ids in documents for t in ids], dtype=np.int32)
        np.save(folder / f"{split}.npy", tokens)
        info[split] = {
            "tokens": len(tokens),
            "sources": counts,
            "document_hashes": [h for h, _ in documents],
            "tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        }
    assert not set(info["train"]["document_hashes"]) & set(info["validation"]["document_hashes"])
    (folder / "data.json").write_text(json.dumps(info))
    runs.commit()
    return info


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=5400,
    retries=0,
    max_containers=5,
    volumes={"/runs": runs},
)
def train(run, variant, deadline, steps=800):
    assert time.time() < deadline
    import math

    import numpy as np
    import torch
    from model_hybrid.model import HybridConfig, HybridLM
    from model_hybrid.study_mixers import configure

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    runs.reload()
    folder = Path("/runs") / run
    arrays = {s: np.load(folder / f"{s}.npy") for s in ["train", "validation"]}
    cfg = HybridConfig(
        vocab_size=32768,
        eos_id=2,
        block_size=512,
        n_layer=8,
        n_embd=256,
        n_head=4,
        n_kv_head=2,
        n_kda_head=4,
        head_dim=64,
        mlp_hidden=768,
        loss_chunk_size=1024,
    )

    def batch(split, index):
        arr = arrays[split]
        start = index * 4096
        assert start + 4097 <= len(arr), (split, index, len(arr))
        rows = np.stack([arr[start + i * 512 : start + (i + 1) * 512 + 1] for i in range(8)])
        x = torch.tensor(rows[:, :-1], device="cuda", dtype=torch.long)
        y = torch.tensor(rows[:, 1:], device="cuda", dtype=torch.long)
        y[y == 1] = -100
        return x, y

    results = []
    for seed in [17, 29]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = configure(HybridLM(cfg), variant).cuda()
        opt = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
        )
        curves = []
        elapsed = 0.0
        targets = 0

        def evaluate(model=model):
            model.eval()
            ce = count = 0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(16):
                    x, y = batch("validation", i)
                    out = model(x, y)
                    ce += out["ce"].item()
                    count += out["count"].item()
            model.train()
            return ce / count

        initial = evaluate()
        for step in range(steps):
            if time.time() > deadline - 120:
                raise TimeoutError("study deadline")
            lr = (
                3e-4
                * min(1.0, (step + 1) / 50)
                * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
            )
            for group in opt.param_groups:
                group["lr"] = lr
            x, y = batch("train", step)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, y)
                loss = out["loss"] / out["count"]
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            if step >= 10:
                elapsed += seconds
                targets += out["count"].item()
            if (step + 1) % 100 == 0 or step + 1 == steps:
                row = {
                    "step": step + 1,
                    "train_ce": (out["ce"] / out["count"]).item(),
                    "validation_ce": evaluate(),
                    "tokens_per_second": targets / elapsed,
                }
                curves.append(row)
                print(variant, seed, row, flush=True)
                (folder / f"{variant}-{seed}.json").write_text(
                    json.dumps(
                        {
                            "variant": variant,
                            "seed": seed,
                            "initial_ce": initial,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "curve": curves,
                            "steps_planned": steps,
                        }
                    )
                )
                runs.commit()
        results.append(json.loads((folder / f"{variant}-{seed}.json").read_text()))
        del model, opt
        torch.cuda.empty_cache()
    return results


@app.function(
    image=image, gpu="H100", cpu=8, memory=32768, timeout=1200, retries=0, volumes={"/runs": runs}
)
def benchmark(run, deadline):
    import numpy as np
    import torch
    from model_hybrid.model import HybridConfig, HybridLM
    from model_hybrid.study_mixers import configure
    from model_hybrid.test_study_mixers import checks

    assert time.time() < deadline
    torch.set_num_threads(4)
    checked = checks()
    torch.backends.cuda.matmul.allow_tf32 = True
    runs.reload()
    arr = np.load(Path("/runs") / run / "train.npy")
    rows = np.stack([arr[i * 512 : (i + 1) * 512 + 1] for i in range(8)])
    x = torch.tensor(rows[:, :-1], device="cuda", dtype=torch.long)
    y = torch.tensor(rows[:, 1:], device="cuda", dtype=torch.long)
    y[y == 1] = -100
    cfg = HybridConfig(
        vocab_size=32768,
        eos_id=2,
        block_size=512,
        n_layer=8,
        n_embd=256,
        n_head=4,
        n_kv_head=2,
        n_kda_head=4,
        head_dim=64,
        mlp_hidden=768,
        loss_chunk_size=1024,
    )
    models = {}
    for variant in ["baseline", "none", "rope", "middle_rope", "middle_rope_conv", "rotating"]:
        torch.manual_seed(123)
        model = configure(HybridLM(cfg), variant).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
        models[variant] = (model, optimizer)

    def step(variant):
        model, optimizer = models[variant]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, y)
            loss = out["loss"] / out["count"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for variant in models:
        for _ in range(10):
            step(variant)
    times = {v: [] for v in models}
    for repeat in range(4):
        order = list(models)
        if repeat % 2:
            order.reverse()
        for variant in order:
            assert time.time() < deadline - 60
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                step(variant)
            torch.cuda.synchronize()
            times[variant].append((time.perf_counter() - start) / 20)
    return {
        "checks": checked,
        "gpu": torch.cuda.get_device_name(),
        "seconds_per_step": times,
        "token_positions_per_step": 4096,
        "torch": str(torch.__version__),
    }


@app.function(
    image=image, gpu="H100", cpu=8, memory=32768, timeout=1800, retries=0, volumes={"/runs": runs}
)
def kernel_study(run, deadline):
    import numpy as np
    from model_hybrid.kernel_study import compare_training, qualify

    assert time.time() < deadline
    result = {"correctness": qualify()}
    runs.reload()
    folder = Path("/runs") / run
    (folder / "kernel-correctness.json").write_text(json.dumps(result))
    runs.commit()
    result["training"] = compare_training(np.load(folder / "train.npy"), deadline)
    (folder / "kernel-study.json").write_text(json.dumps(result))
    runs.commit()
    assert result["training"]["max_ce_difference"] < 0.03, result["training"]["max_ce_difference"]
    return result
