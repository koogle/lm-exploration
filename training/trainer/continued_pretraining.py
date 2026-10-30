"""Seed a new data phase from a trainer checkpoint (continued pretraining).

The seed keeps the parent's weights, the complete optimizer state of whichever optimizer
the config builds (AdamW moments and step counts, or Muon momentum buffers plus the
AdamW state of its auxiliary parameters) and every rank's RNG state. It starts fresh data
cursors and resets the phase counters (step 0, token 0), so the learning-rate schedule
(warmup and cosine over `target_tokens`) restarts from the new config. Optimizer
hyperparameters (weight decay, betas, Muon momentum and learning-rate ratio) also come
from the new config. Lineage records the parent and the lifetime target count.

Run it where the parent checkpoint and the new snapshot are reachable, with the same
PyTorch build as the training image, then resume the trainer from the seed. The
workflow launcher does both with `--phase-from` (see `workflows/README.md`).
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def _with_new_hyperparameters(state, fresh):
    """Parent per-parameter state with the new config's parameter groups.

    Works for a torch optimizer state dict and for composites that nest them by name.
    """
    if "state" in fresh and "param_groups" in fresh:
        old, new = state["param_groups"], fresh["param_groups"]
        if [g["params"] for g in old] != [g["params"] for g in new]:
            raise ValueError("Parent optimizer parameter groups differ from the config")
        return {"state": state["state"], "param_groups": new}
    if set(state) != set(fresh):
        raise ValueError("Parent optimizer structure differs from the config")
    return {key: _with_new_hyperparameters(state[key], fresh[key]) for key in fresh}


def _check_optimizer_state(optimizer):
    """Every saved tensor is finite and shaped like its parameter; some state exists."""
    import torch

    entries = 0
    for part in optimizer.parts.values():
        for parameter, state in part.state.items():
            entries += 1
            for name, value in state.items():
                if not torch.is_tensor(value):
                    continue
                if value.ndim and value.shape != parameter.shape:
                    raise ValueError(f"Optimizer {name} shape differs from its parameter")
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite optimizer {name}")
    if not entries:
        raise ValueError("Parent checkpoint has no optimizer state")


def migrate_phase(parent_path, output, manifest_path, config, *, note=""):
    """Write a seed checkpoint for `config` and the snapshot at `manifest_path`."""
    import torch
    from model_100m.model import GPT, ModelConfig
    from training_data.mixture import open_mixture
    from training_data.subword import (
        BOS_TOKEN_ID,
        EOS_TOKEN_ID,
        TOKENIZER_SHA256,
        VOCAB_SIZE,
        digest,
    )

    from trainer.loop import CHECKPOINT_FORMAT
    from trainer.optimizers import make_optimizer
    from trainer.subword import source_hashes

    parent_path, output = Path(parent_path), Path(output)
    if output.exists():
        raise ValueError("Refuse to overwrite an existing phase seed")
    # Only trusted checkpoints written by our own trainer.
    saved = torch.load(parent_path, map_location="cpu", weights_only=False)
    if saved["format"] != CHECKPOINT_FORMAT:
        raise ValueError("Parent is not a trainer checkpoint")
    if saved["data_identity"]["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise ValueError("Parent tokenizer differs from the frozen tokenizer")
    model_config = ModelConfig(
        vocab_size=VOCAB_SIZE,
        block_size=config.context,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
    )
    architecture = {"module": ModelConfig.__module__, "config": asdict(model_config)}
    if saved["architecture"] != architecture:
        raise ValueError("Parent architecture differs from the config")
    if saved["config"]["world_size"] != config.world_size or len(saved["rng"]) != config.world_size:
        raise ValueError("Parent world size differs from the config")
    if saved["config"]["optimizer"] != config.optimizer:
        raise ValueError("Parent optimizer state belongs to a different optimizer")

    model = GPT(model_config)
    # Optimizer state is indexed by parameter order, so the order must match exactly.
    if list(model.state_dict()) != list(saved["model"]):
        raise ValueError("Parent parameter order differs from the model")
    model.load_state_dict(saved["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Parent weights are not finite")
    optimizer = make_optimizer(model, config)
    state = _with_new_hyperparameters(saved["optimizer"], optimizer.state_dict())
    optimizer.load_state_dict(state)
    _check_optimizer_state(optimizer)

    data = open_mixture(
        manifest_path,
        loader=config.data_loader,
        context=config.context,
        allow_unreviewed=config.allow_unreviewed_data,
    )
    try:
        identity, data_state = data.identity, data.state_dict()
    finally:
        data.close()
    lineage = {
        "parent_checkpoint": str(parent_path),
        "parent_sha256": digest(parent_path),
        "parent_step": saved["step"],
        "parent_phase_tokens": saved["tokens"],
        # Lifetime targets trained before this phase.
        "parent_tokens": saved.get("lineage", {}).get("parent_tokens", 0) + saved["tokens"],
        "parent_config": saved["config"],
        "parent_data_identity": saved["data_identity"],
        "parent_source_sha256": saved["source_sha256"],
        "parent_torch_version": saved["torch_version"],
        "parent_lineage": saved.get("lineage"),
        "note": note,
    }
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": config.as_dict(),
        "architecture": architecture,
        "special_tokens": {"bos": BOS_TOKEN_ID, "eos": EOS_TOKEN_ID},
        "data_identity": identity,
        "source_sha256": source_hashes(),
        "torch_version": str(torch.__version__),
        "step": 0,
        "tokens": 0,
        "model": saved["model"],
        "optimizer": state,
        "data": data_state,
        "rng": saved["rng"],
        "lineage": lineage,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(output)
    return lineage


if __name__ == "__main__":
    from distributed.config import DistributedConfig

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parent", required=True, help="Parent step-NNNNNNNNN.pt")
    parser.add_argument("--output", required=True, help="Seed checkpoint to write")
    parser.add_argument("--manifest", required=True, help="New phase's snapshot manifest")
    parser.add_argument("--config", required=True, help="New phase's DistributedConfig JSON")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    config = DistributedConfig(**json.loads(Path(args.config).read_text()))
    lineage = migrate_phase(args.parent, args.output, args.manifest, config, note=args.note)
    summary = ("parent_checkpoint", "parent_step", "parent_phase_tokens", "parent_tokens")
    print(json.dumps({key: lineage[key] for key in summary}, indent=2))
