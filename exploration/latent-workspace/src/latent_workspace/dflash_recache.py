"""Add fixed feature depths to archived teacher sequences without generating labels."""

import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from .dflash import features
from .dflash_scaling import teacher_statistics
from .train import atomic_save, write_json


def sequence_digest(data):
    digest = hashlib.sha256()
    for split, examples in data.items():
        for e in examples:
            digest.update(split.encode() + e["group"].encode() + e["tokens"].numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def add_features(target, data, old_layers, new_layers, batch_size, deadline):
    if set(data) != {"train", "validation"}:
        raise ValueError("Only training and validation caches are permitted")
    if not set(old_layers) <= set(new_layers):
        raise ValueError("Feature expansion must retain all original layers")
    before = sequence_digest(data)
    added = [layer for layer in new_layers if layer not in old_layers]
    if not added:
        raise ValueError("No added feature layers")
    h = target.config.hidden_size
    for rows in data.values():
        for start in range(0, len(rows), batch_size):
            if time.monotonic() >= deadline:
                raise TimeoutError("Feature recache deadline")
            batch = rows[start : start + batch_size]
            length = max(len(e["tokens"]) for e in batch)
            tokens = torch.zeros(len(batch), length, dtype=torch.long, device=target.device)
            mask = torch.zeros_like(tokens)
            for i, e in enumerate(batch):
                tokens[i, : len(e["tokens"])] = e["tokens"].to(target.device)
                mask[i, : len(e["tokens"])] = 1
            output = target.model(
                tokens, attention_mask=mask, use_cache=False, output_hidden_states=True
            )
            fresh = features(output, added).cpu()
            for i, e in enumerate(batch):
                if e["memory"].shape[-1] != h * len(old_layers):
                    raise ValueError("Original feature width mismatch")
                chunks = dict(zip(old_layers, e["memory"].split(h, dim=-1)))
                chunks.update(zip(added, fresh[i, : len(e["tokens"])].split(h, dim=-1)))
                e["memory"] = torch.cat([chunks[layer] for layer in new_layers], dim=-1)
    if sequence_digest(data) != before:
        raise AssertionError("Recache changed token identities")
    return before


def recache_features(cfg, parent, output, on_checkpoint=None):
    cfg.validate()
    started = time.monotonic()
    parent, output = Path(parent), Path(output)
    source = json.loads((parent / "report.json").read_text())
    if not source.get("completed"):
        raise ValueError("Parent must be complete")
    old_cfg = source["config"]
    for key in ("model", "model_revision", "prompt_tokens", "response_tokens", "block_size"):
        if getattr(cfg, key) != old_cfg[key]:
            raise ValueError(f"Recache changes teacher identity: {key}")
    if cfg.teacher_prompt_style != old_cfg.get("teacher_prompt_style", "raw"):
        raise ValueError("Recache changes teacher prompt style")
    output.mkdir(parents=True, exist_ok=False)
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model, revision=cfg.model_revision, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    data = torch.load(parent / "teacher.pt", map_location="cpu", weights_only=True)
    identity = add_features(
        target,
        data,
        old_cfg["layers"],
        cfg.layers,
        cfg.teacher_batch_size,
        started + cfg.max_seconds - 120,
    )
    atomic_save(output / "teacher.pt", data)
    shutil.copyfile(parent / "prompts.json", output / "prompts.json")
    write_json(output / "config.json", cfg.to_dict())
    audit = teacher_statistics(data, cfg.block_size)
    audit["recache_parent"] = str(parent)
    write_json(output / "teacher-audit.json", audit)
    report = {
        "completed": True,
        "config": cfg.to_dict(),
        "parent": str(parent),
        "teacher_sequence_sha256": identity,
        "old_feature_layers": old_cfg["layers"],
        "new_feature_layers": cfg.layers,
        "preserved_old_features_bitwise": True,
        "teacher_audit": audit,
        "elapsed_seconds": time.monotonic() - started,
        "test_status": "No test tokens, features or responses accessed",
    }
    source_hash = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(file.name.encode() + file.read_bytes())
    report["source_sha256"] = source_hash.hexdigest()
    write_json(output / "report.json", report)
    if on_checkpoint:
        on_checkpoint()
    return report
