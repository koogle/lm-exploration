"""Recipe validation and the decoding loop shared by the Modal launcher and CPU tests."""

import time

import torch

from sampling.recipe import MODE_DEFAULTS, expand_modes, prompts


def configure_precision(precision):
    """FP32 disables TF32 everywhere; BF16 autocast leaves PyTorch defaults unchanged."""
    if precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    return {
        "precision": precision,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def next_token(logits, context, mode):
    """Penalty, temperature, top-k then nucleus; greedy when temperature is zero."""
    penalty = mode["repetition_penalty"]
    if penalty != 1.0:
        # Sign-aware, once per token identity already in the prompt or generated prefix.
        seen = context.unique()
        old = logits[seen]
        logits[seen] = torch.where(old < 0, old * penalty, old / penalty)
    if mode["temperature"] == 0:
        return int(logits.argmax())
    scaled = logits / mode["temperature"]
    if mode["top_k"]:
        values, indices = torch.topk(scaled, mode["top_k"])
    else:
        values, indices = torch.sort(scaled, descending=True)
    if mode["top_p"] < 1:
        # Keep the token that crosses the threshold.
        remove = values.softmax(-1).cumsum(-1) > mode["top_p"]
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        values[remove] = -float("inf")
    return int(indices[torch.multinomial(values.softmax(-1), 1)])


def generate(forward, prefix, mode, seed, max_new_tokens, eos_token_id, device, autocast=False):
    """forward(tokens [1, T]) returns logits [1, T, vocab]; stops at EOS or the token cap."""
    torch.manual_seed(seed)
    device = torch.device(device)
    tokens = torch.tensor([prefix], device=device)
    generated = []
    for _ in range(max_new_tokens):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            logits = forward(tokens)[0, -1].float()
        token = next_token(logits, tokens[0], mode)
        generated.append(token)
        tokens = torch.cat([tokens, torch.tensor([[token]], device=device)], dim=1)
        if token == eos_token_id:
            break
    return generated


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing further sampling work")


@torch.inference_mode()
def sample_model(recipe, model, deadline):
    """All prompt x mode x seed samples for one loaded model.

    `model` provides forward, encode(prompt) -> prefix ids, decode(ids), eos_token_id, device.
    """
    samples = []
    autocast = recipe["precision"] == "bf16-autocast"
    for index, (prompt, group) in enumerate(prompts(recipe)):
        prefix = model.encode(prompt)
        for mode in expand_modes(recipe):
            check_deadline(deadline)
            seed = mode["seed_offset"] + index
            ids = generate(
                model.forward,
                prefix,
                mode,
                seed,
                recipe["max_new_tokens"],
                model.eos_token_id,
                model.device,
                autocast,
            )
            samples.append(
                {
                    "prompt": prompt,
                    "prompt_group": group,
                    "mode": mode["name"],
                    **{key: mode[key] for key in MODE_DEFAULTS},
                    "seed": seed,
                    "generated_tokens": len(ids),
                    "ended_with_eos": ids[-1] == model.eos_token_id,
                    "generation_ids": ids,
                    "completion": model.decode(ids),
                }
            )
    return samples


def run_recipe(recipe, open_runs, deadline, existing=None, save=None):
    """Sample every run that open_runs(model_spec, done_names) yields as (info, model).

    Completed runs in `existing` are reused only when its recipe is identical.
    `save(report)` is called after each run so partial results survive a timeout.
    """
    started = time.monotonic()
    resumed = existing if existing and existing.get("recipe") == recipe else None
    report = {
        "recipe": recipe,
        "torch_version": str(torch.__version__),
        **configure_precision(recipe["precision"]),
        "models": list(resumed["models"]) if resumed else [],
    }
    previous_elapsed = resumed["elapsed_seconds"] if resumed else 0.0
    done = {run["name"] for run in report["models"]}
    for spec in recipe["models"]:
        for info, model in open_runs(spec, done):
            check_deadline(deadline)
            report["models"].append({**info, "samples": sample_model(recipe, model, deadline)})
            report["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
            if save:
                save(report)
    report["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
    return report
