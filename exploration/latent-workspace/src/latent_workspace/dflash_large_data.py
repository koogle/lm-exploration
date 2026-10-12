"""Resumable token-only teacher preparation for the response-length/data-size grid."""

import hashlib
import json
import time
from pathlib import Path

import torch

from .dflash_data import load_teacher_cache, token_hash
from .dflash_scaling import diverse_manifest, generate_batch, teacher_statistics
from .train import atomic_save, seed_all, write_json


def prepare_manifest(cfg, output, parent, on_checkpoint=None):
    from transformers import AutoTokenizer

    cfg.validate()
    output, parent = Path(output), Path(parent)
    output.mkdir(parents=True, exist_ok=False)
    historical = torch.load(cfg.historical_corpus, map_location="cpu", weights_only=True)[
        "metadata"
    ]
    prior = json.loads(Path("/runs/dflash-pilot-02/prompts.json").read_text())
    retained = json.loads((parent / "prompts.json").read_text())
    prior["records"].extend(retained["records"])
    manifest = diverse_manifest(
        cfg,
        AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision),
        output,
        time.monotonic() + cfg.max_seconds - 120,
        historical,
        prior,
        retained["records"],
    )
    manifest["parent_manifest"] = str(parent)
    write_json(output / "prompts.json", manifest)
    write_json(output / "config.json", cfg.to_dict())
    if on_checkpoint:
        on_checkpoint()
    return {
        "completed": True,
        "records": len(manifest["records"]),
        "source_group_counts": manifest["source_group_counts"],
        "test_status": manifest["test_status"],
    }


def extend_response(target, tokenizer, record, previous, response_tokens):
    """Preserve every old teacher token; extend only nonterminated trajectories."""
    if previous is None:
        return generate_batch(target, [record], response_tokens, tokenizer.eos_token_id)[0]
    tokens = previous["tokens"].tolist()
    if (
        tokens[-1] == tokenizer.eos_token_id
        or len(tokens) >= previous["prompt_length"] + response_tokens
    ):
        return tokens
    remaining = previous["prompt_length"] + response_tokens - len(tokens)
    return generate_batch(target, [{"prompt_ids": tokens}], remaining, tokenizer.eos_token_id)[0]


def prepare_chunk(cfg, output, manifest_parent, seed_parent, resume_parent="", on_checkpoint=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg.validate()
    if not cfg.online_teacher_features:
        raise ValueError("large teacher pool requires online features")
    output, manifest_parent, seed_parent = Path(output), Path(manifest_parent), Path(seed_parent)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    seed_all(cfg.seed)
    manifest = json.loads((manifest_parent / "prompts.json").read_text())
    manifest_hash = hashlib.sha256((manifest_parent / "prompts.json").read_bytes()).hexdigest()
    seed_cfg = json.loads((seed_parent / "config.json").read_text())
    manifest_cfg = json.loads((manifest_parent / "config.json").read_text())
    for key in ("model", "model_revision", "layers", "prompt_tokens", "block_size"):
        if json.dumps(cfg.to_dict()[key]) != json.dumps(seed_cfg[key]) or json.dumps(
            cfg.to_dict()[key]
        ) != json.dumps(manifest_cfg[key]):
            raise ValueError(f"large preparation changes teacher identity: {key}")
    original = load_teacher_cache(seed_parent / "teacher.pt")
    old_train = {e["group"]: e for e in original["train"]}
    if resume_parent:
        parent = Path(resume_parent)
        state = json.loads((parent / "progress.json").read_text())
        if state["manifest_sha256"] != manifest_hash or json.dumps(
            state["config"], sort_keys=True
        ) != json.dumps(cfg.to_dict(), sort_keys=True):
            raise ValueError("teacher continuation changes manifest or configuration")
        data = load_teacher_cache(parent / "teacher.pt")
        cursor, seen = state["cursor"], set(state["response_hashes"])
        short, duplicates = state["short"], state["duplicates"]
    else:
        data = {"train": [], "validation": original["validation"]}
        cursor, short, duplicates = 0, 0, 0
        seen = {e["response_sha256"] for e in data["validation"]}
    from .corpus import DOMAINS

    counts = {domain: 0 for domain in DOMAINS}
    rows = []
    for record in manifest["records"]:
        if record["split"] == "train" and counts[record["domain"]] < cfg.train_prompts // 3:
            rows.append(record)
            counts[record["domain"]] += 1
    if any(count != cfg.train_prompts // 3 for count in counts.values()):
        raise ValueError("manifest lacks requested balanced training prompts")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model, revision=cfg.model_revision, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )

    def save():
        atomic_save(output / "teacher.pt", data)
        write_json(
            output / "progress.json",
            {
                "cursor": cursor,
                "manifest_sha256": manifest_hash,
                "config": cfg.to_dict(),
                "response_hashes": sorted(seen),
                "short": short,
                "duplicates": duplicates,
            },
        )
        write_json(output / "config.json", cfg.to_dict())
        write_json(output / "prompts.json", manifest)
        audit = teacher_statistics(data, cfg.block_size)
        audit.update(
            short_responses=short,
            duplicate_responses=duplicates,
            online_training_features=True,
            validation_response_cap=64,
        )
        write_json(output / "teacher-audit.json", audit)
        report = {
            "completed": cursor == len(rows),
            "cursor": cursor,
            "requested": len(rows),
            "config": cfg.to_dict(),
            "teacher_audit": audit,
            "elapsed_seconds": time.monotonic() - started,
            "manifest_sha256": manifest_hash,
            "test_status": "test prompts only; no generated test labels/features/metrics",
            "note": "Old teacher prefixes retained exactly; training tokens only; unchanged 64-token validation cache.",
        }
        write_json(output / "report.json", report)
        if on_checkpoint:
            on_checkpoint()
        return report

    while cursor < len(rows) and time.monotonic() < started + cfg.max_seconds - 180:
        records = rows[cursor : cursor + cfg.teacher_batch_size]
        # Batch together only records requiring the same generation length.
        pending = {}
        sequences = {}
        for i, record in enumerate(records):
            previous = old_train.get(record["group"])
            tokens = previous["tokens"].tolist() if previous is not None else record["prompt_ids"]
            remaining = cfg.response_tokens - (len(tokens) - len(record["prompt_ids"]))
            if previous is not None and (tokens[-1] == tokenizer.eos_token_id or remaining <= 0):
                sequences[i] = tokens
            else:
                pending.setdefault(remaining, []).append((i, {"prompt_ids": tokens}))
        for remaining, batch in pending.items():
            generated = generate_batch(
                target, [r for _, r in batch], remaining, tokenizer.eos_token_id
            )
            sequences.update({i: tokens for (i, _), tokens in zip(batch, generated)})
        for i, record in enumerate(records):
            tokens = sequences[i]
            response = tokens[len(record["prompt_ids"]) :]
            response_hash = token_hash(response)
            if len(response) < cfg.block_size:
                short += 1
            elif response_hash in seen:
                duplicates += 1
            else:
                seen.add(response_hash)
                data["train"].append(
                    {
                        **record,
                        "tokens": torch.tensor(tokens),
                        "prompt_length": len(record["prompt_ids"]),
                        "response_sha256": response_hash,
                    }
                )
        cursor += len(records)
        if cursor % (cfg.teacher_batch_size * 16) == 0:
            print(
                f"Long teacher {cursor}/{len(rows)}; usable {len(data['train'])}; {time.monotonic() - started:.1f}s",
                flush=True,
            )
        if cursor % (cfg.teacher_batch_size * 128) == 0:
            save()
    return save()
