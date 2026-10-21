"""Explicit new-phase settings and optimizer-preserving checkpoint migration."""

import math
from fractions import Fraction
from pathlib import Path


def coding_double_mix(sources):
    """Double the coding share; preserve all remaining relative proportions."""
    old = {name: Fraction(str(source["weight_percent"])) for name, source in sources.items()}
    if sum(old.values()) != 100 or "code_python" not in old:
        raise ValueError("Expected a normalized original mix with code_python")
    coding = old["code_python"] * 2
    if coding >= 100:
        raise ValueError("Doubling coding must leave room for other sources")
    factor = (100 - coding) / (100 - old["code_python"])
    return {
        name: coding if name == "code_python" else share * factor for name, share in old.items()
    }


def token_targets(weights, total=5_000_000_000):
    """Integer allocation with deterministic largest-remainder rounding."""
    if type(total) is not int or total < 1 or sum(weights.values()) != 100:
        raise ValueError("Expected positive total and normalized weights")
    exact = {name: total * share / 100 for name, share in weights.items()}
    targets = {name: math.floor(value) for name, value in exact.items()}
    left = total - sum(targets.values())
    for name in sorted(exact, key=lambda n: (-(exact[n] - targets[n]), n))[:left]:
        targets[name] += 1
    return targets


def capacity_plan(sources, prior_exposures, total=5_000_000_000):
    weights = coding_double_mix(sources)
    targets = token_targets(weights, total)
    return {
        name: {
            "weight_percent_exact": str(weights[name]),
            "weight_percent": float(weights[name]),
            "phase_target_tokens": targets[name],
            "cached_unique_train_tokens": s["splits"]["train"]["loss_tokens"],
            "prior_exposures": prior_exposures.get(name, 0),
            "phase_effective_passes": targets[name] / s["splits"]["train"]["loss_tokens"],
            "lifetime_effective_passes_if_reused": (targets[name] + prior_exposures.get(name, 0))
            / s["splits"]["train"]["loss_tokens"],
            "minimum_new_unique_tokens_for_fresh_two_pass_phase": math.ceil(targets[name] / 2),
        }
        for name, s in sources.items()
    }


def migrate_phase(parent_path, output, manifest_path, config, phase_request):
    """Preserve weights/Adam/RNG; reset phase data and retain cumulative lineage."""
    import torch
    from model_100m.model import GPT, ModelConfig
    from training_data.raw_token_stream import RawTokenMixture
    from training_data.subword import TOKENIZER_SHA256, digest

    from trainer.loop import source_hashes

    parent_path, output = Path(parent_path), Path(output)
    if config.optimizer != "adamw":
        # Only Adam moments are validated and migrated; Muon parents need their own path.
        raise ValueError("Phase migration currently supports AdamW parents only")
    if output.exists():
        raise ValueError("Refuse seed overwrite")
    saved = torch.load(parent_path, map_location="cpu", weights_only=False)
    if (
        saved.get("format") != "scholarly-ddp-v1"
        or saved["data_identity"]["tokenizer_sha256"] != TOKENIZER_SHA256
    ):
        raise ValueError("Checkpoint or tokenizer mismatch")
    for name in ("n_layer", "n_head", "n_embd", "context", "world_size"):
        if saved["config"][name] != getattr(config, name):
            raise ValueError("Migration architecture/GPU mismatch: " + name)
    if saved["torch_version"] != str(torch.__version__) or len(saved["rng"]) != config.world_size:
        raise ValueError("Parent framework or RNG mismatch")
    model = GPT(
        ModelConfig(
            vocab_size=32768,
            block_size=config.context,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
        )
    )
    if list(model.state_dict()) != list(saved["model"]):
        raise ValueError("Parameter order changed; cannot safely migrate Adam moments")
    model.load_state_dict(saved["model"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite model")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.peak_lr, betas=(0.9, 0.95), weight_decay=config.weight_decay
    )
    optimizer.load_state_dict(saved["optimizer"])
    if not optimizer.state:
        raise ValueError("Missing parent optimizer")
    for parameter, state in optimizer.state.items():
        for name in ("exp_avg", "exp_avg_sq"):
            if state[name].shape != parameter.shape or not torch.isfinite(state[name]).all():
                raise ValueError("Invalid Adam moments")
    data = RawTokenMixture(manifest_path, context=config.context, allow_unreviewed=True)
    prior_lineage = saved.get("lineage", {})
    lifetime = prior_lineage.get("parent_tokens", 0) + saved["tokens"]
    lineage = {
        "parent_checkpoint": str(parent_path),
        "parent_sha256": digest(parent_path),
        "parent_tokens": lifetime,
        "immediate_parent_phase_tokens": saved["tokens"],
        "immediate_parent_step": saved["step"],
        "parent_source_sha256": saved["source_sha256"],
        "prior_lineage": prior_lineage,
        "phase_request": phase_request,
        "optimizer": "Adam moments/update counts preserved",
        "schedule": {
            "peak_lr": config.peak_lr,
            "min_lr_ratio": config.min_lr_ratio,
            "warmup_tokens": config.warmup_tokens,
            "counter": "phase loss tokens",
        },
    }
    payload = {
        **saved,
        "config": config.as_dict(),
        "source_sha256": source_hashes(),
        "data_identity": data.identity,
        "data": data.state_dict(),
        "step": 0,
        "tokens": 0,
        "lineage": lineage,
    }
    data.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(output)
    return lineage
