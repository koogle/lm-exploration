"""Content-only contrasts with input/output protection for tool-call structure."""

import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from .study import command_metrics, load


def basis(rows, rank):
    rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.linalg.svd(rows, full_matrices=False).Vh[:rank]


def perpendicular(vector, directions):
    return vector - directions.T @ (directions @ vector)


@torch.inference_mode()
def build(model_path, prompts_path, out, deadline):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(model_path)
    examples = [x for x in json.loads(Path(prompts_path).read_text()) if x["group"] == "loops"][:16]
    stats = {layer: {"loop": [], "novel": [], "structure": []} for layer in (14, 17)}
    current, handles = {}, []
    for layer in stats:

        def hook(module, args, output, layer=layer):
            states = args[0][
                0, current["start"] - 1 : current["start"] - 1 + current["length"]
            ].float()
            command = states[current["command"]].mean(0).cpu()
            stats[layer][current["kind"]].append(command)
            for mask in (current["opening"], current["closing"]):
                if any(mask):
                    stats[layer]["structure"].append(states[mask].mean(0).cpu())

        handles.append(model.model.layers[layer].mlp.down_proj.register_forward_hook(hook))
    try:
        for example in examples:
            if time.time() > deadline - 120:
                raise TimeoutError("Contrast deadline")
            target = example["target"]
            encoded = json.dumps(example["command"], ensure_ascii=False)
            lo = target.index(encoded, target.index('"command":')) + 1
            hi = lo + len(encoded) - 2
            tokens = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
            offsets = tokens["offset_mapping"]
            current.update(
                length=len(tokens["input_ids"]),
                command=[b > lo and a < hi for a, b in offsets],
                opening=[b <= lo for a, b in offsets],
                closing=[a >= hi for a, b in offsets],
            )
            for kind, key in (("loop", "prompt"), ("novel", "novel_prompt")):
                prefix = tokenizer.encode(example[key], add_special_tokens=False)
                current.update(kind=kind, start=len(prefix))
                model(
                    input_ids=torch.tensor([prefix + tokens["input_ids"]], device="cuda"),
                    use_cache=False,
                    logits_to_keep=1,
                )
    finally:
        for handle in handles:
            handle.remove()
    specs = [
        {"name": "content64-s2", "columns": 64, "strength": 2, "input_rank": 0},
        {"name": "gated128-s2", "columns": 128, "strength": 2, "input_rank": 16},
        {"name": "gated128-s4", "columns": 128, "strength": 4, "input_rank": 16},
        {"name": "gated64-s4", "columns": 64, "strength": 4, "input_rank": 16},
    ]
    originals = {
        f"model.layers.{layer}.mlp.down_proj.weight": model.model.layers[layer]
        .mlp.down_proj.weight.detach()
        .cpu()
        .clone()
        for layer in stats
    }
    save_file(originals, out / "original-patch.safetensors")
    from safetensors import safe_open

    with safe_open(
        out.parent / "qwen06-loop-20261103-01/candidate-l14-l17-s4/model.safetensors",
        framework="pt",
    ) as handle:
        save_file(
            {key: handle.get_tensor(key) for key in originals}, out / "first-round.safetensors"
        )
    audit = []
    for spec in specs:
        weights, details = {}, []
        for layer, values in stats.items():
            key = f"model.layers.{layer}.mlp.down_proj.weight"
            w = originals[key].float()
            loop, novel = torch.stack(values["loop"]).mean(0), torch.stack(values["novel"]).mean(0)
            delta = loop - novel
            structure = torch.stack(values["structure"])
            direction = perpendicular(w @ delta, basis(structure @ w.T, 8))
            direction /= direction.norm().clamp_min(1e-8)
            columns = ((direction @ w) * delta).topk(spec["columns"]).indices
            selected = w[:, columns]
            controls = torch.cat([structure, torch.stack(values["novel"])])[:, columns]
            if spec["input_rank"]:
                protected = basis(controls, spec["input_rank"])
                contrast = perpendicular(delta[columns], protected)
                denominator = contrast @ delta[columns]
                if denominator < 1e-8:
                    raise ValueError("Protected contrast has no usable signal")
                right = contrast / denominator
                amplitude = (direction @ selected) @ delta[columns]
                update = -spec["strength"] * amplitude * direction[:, None] * right[None, :]
            else:
                update = -spec["strength"] * direction[:, None] * (direction @ selected)[None, :]
            # Avoid amplifying a tiny residual contrast into an extreme weight change.
            cap = min(1.0, float(0.25 * selected.norm() / update.norm().clamp_min(1e-8)))
            update *= cap
            edited = originals[key].clone()
            edited[:, columns] = (selected + update).to(edited.dtype)
            weights[key] = edited
            details.append(
                {
                    "layer": layer,
                    "columns": columns.tolist(),
                    "cap_multiplier": cap,
                    "relative_selected_weight_change": float(update.norm() / selected.norm()),
                    "control_write_rms": float((controls @ update.T).square().mean().sqrt()),
                    "changed_scalars": int((edited != originals[key]).sum()),
                }
            )
        save_file(weights, out / f"{spec['name']}.safetensors")
        audit.append({**spec, "layers": details})
    (out / "build.json").write_text(json.dumps(audit, indent=2))
    # Read the installed engine API before using its live weight-loading interface.
    from vllm import LLM

    print("COLLECTIVE_API", inspect.signature(LLM.collective_rpc), flush=True)
    print(inspect.getsource(LLM.collective_rpc), flush=True)
    print(json.dumps(audit), flush=True)
    return audit


def load_worker_weights(worker, path):
    """vLLM RPC: replace complete down-projection tensors while no requests are live."""
    from safetensors.torch import load_file

    weights = load_file(path)
    loaded = worker.model_runner.model.load_weights(weights.items())
    return sorted(loaded)


def evaluate(model_path, prompts_path, out, deadline, selection):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out = Path(out)
    examples = json.loads(Path(prompts_path).read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    engine = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=40960,
        hf_overrides={"max_position_embeddings": 40960},
        enforce_eager=True,
        enable_prefix_caching=False,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,
        seed=20261104,
    )
    prompts = [
        {"prompt_token_ids": tokenizer.encode(x["prompt"], add_special_tokens=False)}
        for x in examples
    ]
    params = [
        SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens=1536,
            stop_token_ids=[151645, 151643],
            skip_special_tokens=False,
            seed=20261104 + index,
        )
        for index in range(len(examples))
    ]
    original = load_file(out / "original-patch.safetensors")
    specs = selection["candidates"]
    summaries = {}
    first_original = None
    for spec in specs:
        if time.time() > deadline - 120:
            raise TimeoutError("Generation deadline")
        if spec.get("source_patch"):
            source = load_file(out / spec["source_patch"])
            weights = {
                key: (value.float() + spec["scale"] * (source[key].float() - value.float())).to(
                    value.dtype
                )
                for key, value in original.items()
            }
            save_file(weights, out / spec["patch"])
        else:
            weights = original if spec["name"] == "original" else load_file(out / spec["patch"])
        patch_path = out / (
            "original-patch.safetensors" if spec["name"] == "original" else spec["patch"]
        )
        loaded = engine.collective_rpc(load_worker_weights, args=(str(patch_path),))
        if not all(set(names) == set(weights) for names in loaded):
            raise ValueError(f"Incomplete weight load: {loaded}")
        outputs = engine.generate(prompts, params, use_tqdm=False)
        rows = []
        for example, output in zip(examples, outputs, strict=True):
            completion = output.outputs[0]
            rows.append(
                {
                    "id": example["id"],
                    "group": example["group"],
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                    "tokens": len(completion.token_ids),
                    "token_ids": list(completion.token_ids),
                    **command_metrics(completion.text, example["seen_commands"]),
                }
            )
        (out / f"generation-{spec['name']}.json").write_text(json.dumps(rows, indent=2))
        totals = {}
        for group in ("loops", "clean", "format_regression"):
            group_rows = [x for x in rows if x["group"] == group]
            totals[group] = {
                "n": len(group_rows),
                **{
                    key: sum(x[key] for x in group_rows)
                    for key in (
                        "valid",
                        "repeated",
                        "near_repeat",
                        "internal_repeat",
                        "valid_distinct",
                        "premature_submit",
                    )
                },
                "length_limit": sum(x["finish_reason"] == "length" for x in group_rows),
            }
        summaries[spec["name"]] = totals
        print(json.dumps({spec["name"]: totals}), flush=True)
        if spec["name"] == "original":
            first_original = [x["token_ids"] for x in rows]
    # A deterministic restore check catches stale caches or ineffective weight loading.
    engine.collective_rpc(load_worker_weights, args=(str(out / "original-patch.safetensors"),))
    restored = engine.generate(prompts, params, use_tqdm=False)
    same = (
        first_original is not None
        and [list(x.outputs[0].token_ids) for x in restored] == first_original
    )
    if not same:
        raise ValueError("Original output changed after restoring original weights")
    summaries["restore_check"] = {"identical_original_generations": same, "n": len(examples)}
    (out / "generation-summary.json").write_text(json.dumps(summaries, indent=2))
    return summaries


def export(model_path, out, selection):
    from safetensors import safe_open
    from safetensors.torch import load_file

    source, out = Path(model_path), Path(out)
    patch = load_file(out / selection["patch"])
    destination = out / selection["export"]
    if destination.exists():
        raise ValueError("Export destination must be fresh")
    destination.mkdir()
    with safe_open(source / "model.safetensors", framework="pt") as handle:
        tensors = {k: handle.get_tensor(k) for k in handle.keys()}  # noqa: SIM118
        expected = {f"model.layers.{layer}.mlp.down_proj.weight" for layer in (14, 17)}
        if set(patch) != expected:
            raise ValueError("Unexpected patch tensor keys")
        changes = {}
        for key, value in patch.items():
            if value.shape != tensors[key].shape or value.dtype != tensors[key].dtype:
                raise ValueError(f"Patch shape or dtype mismatch: {key}")
            changes[key] = int((value != tensors[key]).sum())
        tensors.update(patch)
        save_file(tensors, destination / "model.safetensors", metadata=handle.metadata())
    import shutil

    for file in source.iterdir():
        if file.suffix in (".json", ".jinja"):
            shutil.copy2(file, destination / file.name)
    manifest = {
        "selection": selection,
        "status": "Experimental candidate; assess live results before use",
        "changed_scalars_by_tensor": changes,
        "patch_sha256": hashlib.sha256((out / selection["patch"]).read_bytes()).hexdigest(),
    }
    with (source / "model.safetensors").open("rb") as handle:
        manifest["source_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    with (destination / "model.safetensors").open("rb") as handle:
        manifest["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    (destination / "weight-edit.json").write_text(json.dumps(manifest, indent=2))
    return manifest
