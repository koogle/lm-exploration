"""Small main-branch KDA model, checkpointed pretraining and native-state probes."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from training_data.packed_shards import PreparedMixture, heldout_index

from trainer.optimizers import OptimizerConfig, RoleOptimizer, set_learning_rate

from .concept_probes import atlas, causal_probes, qualify_recurrence, save_json
from .model import HybridConfig, HybridLM
from .tokenizer import load

CONFIG = {
    "vocab_size": 154880,
    "eos_id": 154820,
    "block_size": 1024,
    "n_layer": 12,
    "n_embd": 448,
    "n_head": 7,
    "n_kv_head": 1,
    "n_kda_head": 7,
    "head_dim": 64,
    "mlp_hidden": 1152,
    "mtp": False,
    "loss_chunk_size": 512,
}


def run(folder, volume, deadline, provenance, steps=12208):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.recompile_limit = 128
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    save_json(folder / "qualification.json", qualify_recurrence(), volume)
    cfg = HybridConfig(**CONFIG)
    model = HybridLM(cfg).cuda()
    optimizer = RoleOptimizer(model, OptimizerConfig(), fused=True)
    recipe = json.loads((Path(__file__).parent / "recipes/hybrid-40b-v1.json").read_text())
    recipe["context"] = cfg.block_size
    stream = PreparedMixture(recipe, "/data/" + recipe["name"], wait_seconds=30)
    val = PreparedMixture(recipe, "/data/" + recipe["name"], wait_seconds=30)
    val.orders = {n: [heldout_index(recipe["sources"][n])] for n in val.names}
    validation = [val.next_batch(1) for _ in range(12)]
    # Native atlas uses separate held-out windows, beyond validation's prefix.
    windows = []
    for _ in range(24):
        xx, _, names = val.next_batch(1)
        windows.append((names[0], xx[0, :256].tolist()))
    tok = load("/project/glm-tokenizer.json")
    result = {
        "status": "running",
        "provenance": provenance,
        "config": asdict(cfg),
        "parameters": sum(p.numel() for p in model.parameters()),
        "seed": 17,
        "planned_tokens": steps * 8192,
        "curve": [],
        "optimizer": "main RoleOptimizer Muon+AdamW",
    }
    start = time.time()

    def evaluate():
        model.eval()
        ce, count = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y, _ in validation:
                out = model(torch.as_tensor(x, device="cuda"), torch.as_tensor(y, device="cuda"))
                ce += out["ce"].item()
                count += out["count"].item()
        model.train()
        return ce / count

    def checkpoint(step):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(cfg),
                "step": step,
                "tokens": step * 8192,
                "data_state": stream.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "provenance": provenance,
            },
            folder / "checkpoint.tmp",
        )
        (folder / "checkpoint.tmp").replace(folder / "checkpoint.pt")
        volume.commit()

    # Untrained control runs before any optimization; preserve random state afterwards.
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    (folder / "untrained").mkdir(exist_ok=True)
    causal_probes(model, tok, folder / "untrained", volume, deadline, quick=True)
    atlas(model, tok, windows[:6], folder / "untrained", volume, deadline)
    torch.set_rng_state(state)
    torch.cuda.set_rng_state_all(cuda_state)
    result["initial_ce"] = evaluate()
    save_json(folder / "metrics.json", result, volume)
    checkpoint(0)
    step = 0
    for step in range(1, steps + 1):
        # Reserve at least 90 minutes for final probes and archive.
        if time.time() > deadline - 5400:
            break
        model.train()
        optimizer.zero_grad()
        factor = min(1.0, step / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        set_learning_rate(optimizer, factor)
        x, y, _ = stream.next_batch(8)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(torch.as_tensor(x, device="cuda"), torch.as_tensor(y, device="cuda"))
            loss = out["loss"] / out["count"]
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 256 == 0 or step == steps:
            result["curve"].append(
                {
                    "step": step,
                    "tokens": step * 8192,
                    "validation_ce": evaluate(),
                    "train_ce": float(out["ce"] / out["count"]),
                    "grad_norm": float(norm),
                    "seconds": time.time() - start,
                }
            )
            result["steps_completed"] = step
            save_json(folder / "metrics.json", result, volume)
            print("PROGRESS", result["curve"][-1], flush=True)
        if step % 1024 == 0 or step == 1 or step == steps:
            checkpoint(step)
    else:
        step = steps + 1
    completed = step - 1
    checkpoint(completed)
    result["steps_completed"] = completed
    result["tokens"] = completed * 8192
    result["training_status"] = "complete" if completed == steps else "deadline"
    save_json(folder / "metrics.json", result, volume)
    optimizer = None
    torch.cuda.empty_cache()
    (folder / "trained").mkdir(exist_ok=True)
    atlas(model, tok, windows, folder / "trained", volume, deadline)
    causal_probes(model, tok, folder / "trained", volume, deadline)
    result["status"] = "complete" if completed == steps else "partial_training"
    result["seconds"] = time.time() - start
    save_json(folder / "metrics.json", result, volume)
    return result
