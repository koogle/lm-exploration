"""Paired held-out behavior, interventions, and checkpoint gradient diagnostics."""

import json
import time
from pathlib import Path

from .branch_study import RECIPE, prepared_volume
from .modal_study import app, image, runs

SOURCE_RUN = "kda-glm-1b-1793107241"


def run_diagnostics(run, variant, seed, deadline):
    import hashlib
    from collections import defaultdict

    import numpy as np
    import torch
    from model_hybrid.model import HybridConfig, HybridLM
    from model_hybrid.prepared_data import PreparedMixture, heldout_index, shard_paths
    from model_hybrid.study_mixers import configure
    from torch.nn import functional as F

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    runs.reload()
    folder = Path("/runs") / run / f"{variant}-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    origin = Path("/runs") / SOURCE_RUN / f"{variant}-{seed}"
    saved = torch.load(origin / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert saved["step"] == 122070
    cfg = HybridConfig(**saved["config"])
    model = configure(HybridLM(cfg), variant).cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    opt = torch.optim.AdamW(
        model.parameters(), lr=3e-5, betas=(0.9, 0.95), weight_decay=0.1, fused=True
    )
    opt.load_state_dict(saved["optimizer"])
    result = {
        "variant": variant,
        "seed": seed,
        "source_run": SOURCE_RUN,
        "checkpoint_step": saved["step"],
        "status": "running",
    }

    def save(stage):
        assert time.time() < deadline - 120
        result["stage"] = stage
        (folder / "result.json").write_text(json.dumps(result))
        runs.commit()
        print(variant, seed, stage, flush=True)

    def tensors(x, y):
        return torch.as_tensor(x, device="cuda", dtype=torch.long), torch.as_tensor(
            y, device="cuda", dtype=torch.long
        )

    def losses(x, y):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.hidden_states(x).flatten(0, 1)
            ce, correct = [], []
            for features, labels in zip(h.split(512), y.flatten().split(512)):
                logits = F.linear(features, model.output_weight).float()
                ce.append(F.cross_entropy(logits, labels, reduction="none", ignore_index=-100))
                correct.append(logits.argmax(-1) == labels)
        return torch.cat(ce).view_as(y), torch.cat(correct).view_as(y)

    def mean_ce(batches):
        total = count = 0
        for x, y in batches:
            ce, _ = losses(*tensors(x, y))
            total += ce.sum().item()
            count += int((y != -100).sum())
        return total / count

    recipe = dict(RECIPE, context=1024)
    root = Path("/data") / recipe["name"]
    original = PreparedMixture(recipe, root, wait_seconds=30)
    original.orders = {name: [heldout_index(recipe["sources"][name])] for name in original.names}
    original_batches = [original.next_batch(8)[:2] for _ in range(32)]
    result["original_validation_ce"] = mean_ce(original_batches)
    expected = json.loads((origin / "metrics.json").read_text())["curve"][-1]["validation_ce"]
    assert abs(result["original_validation_ce"] - expected) < 0.002, (
        result["original_validation_ce"],
        expected,
    )
    del original

    # Independent documents, sampled across the held-out shards rather than only their prefix.
    rng = np.random.default_rng(7547)
    xs, ys, document_meta = [], [], []
    for name in sorted(recipe["sources"]):
        bin_path, docs_path, _ = shard_paths(root, name, heldout_index(recipe["sources"][name]))
        lengths = np.load(docs_path)[:, 0]
        offsets = np.r_[0, np.cumsum(lengths)]
        ids = np.memmap(bin_path, mode="r", dtype=np.uint32)
        chosen = rng.choice(np.flatnonzero(lengths >= 257), 128, replace=False)
        for document in chosen:
            size = min(1025, int(lengths[document]))
            start = int(rng.integers(0, max(1, int(lengths[document]) - size + 1)))
            sequence = np.array(
                ids[offsets[document] + start : offsets[document] + start + size], dtype=np.int64
            )
            x, y = np.zeros(1024, dtype=np.int64), np.full(1024, -100, dtype=np.int64)
            x[: size - 1], y[: size - 1] = sequence[:-1], sequence[1:]
            xs.append(x)
            ys.append(y)
            document_meta.append(
                {"source": name, "document": int(document), "offset": start, "length": size - 1}
            )
        del ids
    xs, ys = np.stack(xs), np.stack(ys)
    result["sample_sha256"] = hashlib.sha256(xs.tobytes() + ys.tobytes()).hexdigest()
    np.savez_compressed(folder / "examples.npz", x=xs, y=ys)
    (folder / "documents.json").write_text(json.dumps(document_meta))
    all_loss, all_correct = [], []
    for offset in range(0, len(xs), 4):
        ce, correct = losses(*tensors(xs[offset : offset + 4], ys[offset : offset + 4]))
        all_loss.append(ce.cpu().numpy())
        all_correct.append(correct.cpu().numpy())
    all_loss, all_correct = np.concatenate(all_loss), np.concatenate(all_correct)
    np.savez_compressed(folder / "token_metrics.npz", ce=all_loss, correct=all_correct)
    result["source_ce"] = {
        name: float(
            all_loss[i * 128 : (i + 1) * 128].sum() / (ys[i * 128 : (i + 1) * 128] != -100).sum()
        )
        for i, name in enumerate(sorted(recipe["sources"]))
    }
    save("paired_evaluation")

    # Intervention subset: identical 16 documents from each source for every model.
    selected = np.concatenate([np.arange(i * 128, i * 128 + 16) for i in range(3)])
    batches = [(xs[ix], ys[ix]) for ix in selected.reshape(-1, 4)]
    base_ce = mean_ce(batches)
    result["interventions"] = {"unaltered": base_ce}
    kdas = [block.mixer for block in model.blocks if block.kind == "kda"]
    if variant == "baseline":
        convs = {
            f"{i}.{name}": getattr(layer, name).detach().clone()
            for i, layer in enumerate(kdas)
            for name in ["q_conv", "k_conv", "v_conv"]
        }
        for paths in [("q_conv",), ("k_conv",), ("v_conv",), ("q_conv", "k_conv", "v_conv")]:
            with torch.no_grad():
                for layer in kdas:
                    for name in paths:
                        getattr(layer, name)[..., :-1] = 0
            result["interventions"]["remove_past_" + "+".join(paths)] = mean_ce(batches)
            with torch.no_grad():
                for i, layer in enumerate(kdas):
                    for name in paths:
                        getattr(layer, name).copy_(convs[f"{i}.{name}"])
        for layer in kdas:
            layer.mode = "baseline_bf16"
        result["interventions"]["bf16_recurrence_inputs"] = mean_ce(batches)
        for layer in kdas:
            layer.mode = "baseline"
        result["convolution_weights"] = {
            name: {
                "rms_by_lag_3_to_0": value.square().mean((0, 1)).sqrt().tolist(),
                "mean_by_lag_3_to_0": value.mean((0, 1)).tolist(),
            }
            for name, value in convs.items()
        }
    elif variant == "rope_triton":
        for layer in kdas:
            layer.mode = "none"
        result["interventions"]["disable_rope"] = mean_ce(batches)
        for layer in kdas:
            layer.mode = "rope_triton"

    # Read the same last-token targets with successively shorter prefixes.
    # Only examples with all 1024 valid targets enter; no artificial EOS tokens.
    long = [int(i) for i in selected if document_meta[i]["length"] == 1024]
    result["context_last_token_ce"] = {}
    for context in [4, 16, 64, 256, 1024]:
        values = []
        for offset in range(0, len(long), 4):
            ix = long[offset : offset + 4]
            x, y = tensors(xs[ix, -context:], ys[ix, -context:])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.hidden_states(x)[:, -1]
                logits = F.linear(h, model.output_weight).float()
                values.extend(F.cross_entropy(logits, y[:, -1], reduction="none").cpu().tolist())
        result["context_last_token_ce"][str(context)] = {
            "mean": float(np.mean(values)),
            "per_example": values,
            "indices": long,
        }
    save("interventions")

    def group(name):
        if name.startswith("blocks."):
            parts = name.split(".")
            kind = model.blocks[int(parts[1])].kind
            component = ".".join(parts[2:4]) if parts[2] == "mixer" else parts[2]
            return f"{kind}/{parts[1]}/{component}"
        return name

    # CE gradients on one batch per source, using the combined valid-token mean.
    grad_batches = [(xs[i * 128 : i * 128 + 4], ys[i * 128 : i * 128 + 4]) for i in range(3)]
    gradient_count = sum(int((y != -100).sum()) for _, y in grad_batches)
    model.zero_grad(set_to_none=True)
    for x, y in grad_batches:
        x, y = tensors(x, y)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, y)
            loss = out["ce"] / gradient_count
        loss.backward()
    heldout_grad = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    grad_records, conv_grads = defaultdict(lambda: defaultdict(float)), {}
    for name, param in model.named_parameters():
        g = heldout_grad[name]
        state = opt.state[param]
        m, v = state["exp_avg"], state["exp_avg_sq"]
        step = float(state["step"])
        direction = -(m / (1 - 0.9**step)) / ((v / (1 - 0.95**step)).sqrt() + 1e-8)
        bucket = grad_records[group(name)]
        for key, value in [
            ("weight_sq", param.square().sum()),
            ("gradient_sq", g.square().sum()),
            ("moment_direction_sq", direction.square().sum()),
            ("gradient_dot_moment_direction", (g * direction).sum()),
        ]:
            bucket[key] += value.item()
        bucket["parameters"] += param.numel()
        if "_conv" in name:
            conv_grads[name] = {
                "gradient_rms_by_lag_3_to_0": g.square().mean((0, 1)).sqrt().tolist(),
                "moment_direction_rms_by_lag_3_to_0": direction.square()
                .mean((0, 1))
                .sqrt()
                .tolist(),
            }
    result["heldout_gradient_groups"] = dict(grad_records)
    result["convolution_gradients"] = conv_grads

    # Exact new AdamW step on the next training batch, including z-loss and clipping.
    # This is a fresh diagnostic update, not a reconstruction of historical gradients.
    training = PreparedMixture(recipe, root, wait_seconds=30)
    training.load_state_dict(saved["data_state"])
    tx, ty, names = training.next_batch(8)
    result["next_training_batch"] = {
        "sha256": hashlib.sha256(tx.tobytes() + ty.tobytes()).hexdigest(),
        "sources": names,
    }
    del saved, training
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    held_before = mean_ce(grad_batches)
    train_before = mean_ce([(tx, ty)])
    opt.zero_grad(set_to_none=True)
    x, y = tensors(tx, ty)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, y)
        loss = out["loss"] / out["count"]
    loss.backward()
    raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    groups = defaultdict(lambda: defaultdict(float))
    for name, param in model.named_parameters():
        delta = param.detach() - before[name]
        bucket = groups[group(name)]
        bucket["update_sq"] += delta.square().sum().item()
        bucket["weight_sq"] += before[name].square().sum().item()
        bucket["heldout_gradient_dot_update"] += (heldout_grad[name] * delta).sum().item()
        bucket["clipped_train_gradient_sq"] += param.grad.square().sum().item()
        bucket["parameters"] += param.numel()
    result["fresh_update"] = {
        "raw_gradient_norm": raw_norm.item(),
        "groups": dict(groups),
        "train_ce_before": train_before,
        "train_ce_after": mean_ce([(tx, ty)]),
        "heldout_ce_before": held_before,
        "heldout_ce_after": mean_ce(grad_batches),
        "lr": opt.param_groups[0]["lr"],
    }
    result["status"] = "complete"
    save("complete")
    return result


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=3600,
    retries=0,
    max_containers=6,
    volumes={"/runs": runs, "/data": prepared_volume},
)
def diagnose(run, variant, seed, deadline):
    assert time.time() < deadline - 300
    return run_diagnostics(run, variant, seed, deadline)
