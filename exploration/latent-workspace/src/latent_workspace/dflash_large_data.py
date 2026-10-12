"""Resumable token-only teacher preparation for the response-length/data-size grid."""

import hashlib
import json
import time
from pathlib import Path

import torch

from .dflash import features
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
    write_json(output / "config.json", cfg.to_dict())

    def save_partial(records):
        write_json(
            output / "prompts.json",
            {
                "records": records,
                "sources": historical["sources"],
                "model": cfg.model,
                "model_revision": cfg.model_revision,
                "seed": cfg.seed,
                "completed": False,
                "test_status": "test prompts only; no labels/features/metrics",
            },
        )
        if on_checkpoint:
            on_checkpoint()

    try:
        manifest = diverse_manifest(
            cfg,
            AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision),
            output,
            time.monotonic() + cfg.max_seconds - 120,
            historical,
            prior,
            retained["records"],
            on_partial=save_partial,
        )
    except TimeoutError:
        partial = json.loads((output / "prompts.json").read_text())
        return {
            "completed": False,
            "records": len(partial["records"]),
            "resume_manifest_parent": str(output),
            "test_status": partial["test_status"],
        }
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
        if (
            state.get("seed_parent") != str(seed_parent)
            or state["manifest_sha256"] != manifest_hash
            or json.dumps(state["config"], sort_keys=True)
            != json.dumps(cfg.to_dict(), sort_keys=True)
        ):
            raise ValueError("teacher continuation changes manifest or configuration")
        data = load_teacher_cache(parent / "teacher.pt")
        cursor, seen = state["cursor"], set(state["response_hashes"])
        short, duplicates = state["short"], state["duplicates"]
    else:
        reuse_long = (
            seed_cfg.get("online_teacher_features", False)
            and seed_cfg["response_tokens"] >= cfg.response_tokens
        )
        data = {
            "train": list(original["train"]) if reuse_long else [],
            "validation": original["validation"],
        }
        cursor, short, duplicates = 0, 0, 0
        seen = {e["response_sha256"] for values in data.values() for e in values}
    from .corpus import DOMAINS

    counts = {domain: 0 for domain in DOMAINS}
    rows = []
    for record in manifest["records"]:
        if record["split"] == "train" and counts[record["domain"]] < cfg.train_prompts // 3:
            rows.append(record)
            counts[record["domain"]] += 1
    rows.sort(key=lambda record: record["group"] not in old_train)
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

    if not resume_parent:
        # Extend the same held-out trajectories for the common long evaluation.
        # Keep the historical short view as an additional diagnostic, never train on either.
        data["validation_short"] = original.get("validation_short", data["validation"])
        long_validation = []
        for previous in original["validation"]:
            sequence = extend_response(target, tokenizer, previous, previous, cfg.response_tokens)
            tokens = torch.tensor(sequence, device=target.device)
            with torch.no_grad():
                memory = (
                    previous["memory"]
                    if sequence == previous["tokens"].tolist()
                    else features(
                        target.model(tokens[None], use_cache=False, output_hidden_states=True),
                        cfg.layers,
                    )[0].cpu()
                )
            response_hash = token_hash(sequence[previous["prompt_length"] :])
            seen.add(response_hash)
            long_validation.append(
                {
                    **previous,
                    "tokens": tokens.cpu(),
                    "memory": memory,
                    "response_sha256": response_hash,
                }
            )
        data["validation"] = long_validation

    def save():
        atomic_save(output / "teacher.pt", data)
        write_json(
            output / "progress.json",
            {
                "cursor": cursor,
                "seed_parent": str(seed_parent),
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
            validation_response_cap=cfg.response_tokens,
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
            "note": "Old teacher prefixes retained exactly; training tokens only; extended validation retains original tokens; historical short view also saved.",
        }
        write_json(output / "report.json", report)
        if on_checkpoint:
            on_checkpoint()
        return report

    reused_groups = {e["group"] for e in data["train"]}
    while cursor < len(rows) and time.monotonic() < started + cfg.max_seconds - 180:
        batch_records = rows[cursor : cursor + cfg.teacher_batch_size]
        records = [r for r in batch_records if r["group"] not in reused_groups]
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
        cursor += len(batch_records)
        if cursor % (cfg.teacher_batch_size * 16) == 0:
            print(
                f"Long teacher {cursor}/{len(rows)}; usable {len(data['train'])}; {time.monotonic() - started:.1f}s",
                flush=True,
            )
        if cursor % (cfg.teacher_batch_size * 128) == 0:
            save()
    return save()


def seal_subset(parent, required_parent, output, count=49152):
    """Seal an available balanced slice without calling an incomplete full pool complete."""
    from .dflash_scaling import nested_subset

    parent, required_parent, output = Path(parent), Path(required_parent), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    progress = json.loads((parent / "progress.json").read_text())
    source_data = load_teacher_cache(parent / "teacher.pt")
    required = load_teacher_cache(required_parent / "teacher.pt")["train"]
    available = {e["group"]: e for e in source_data["train"]}
    for e in required:
        if e["group"] not in available or not torch.equal(
            e["tokens"], available[e["group"]]["tokens"][: len(e["tokens"])]
        ):
            raise ValueError("required teacher prefix is missing or changed")
    data = {
        **source_data,
        "train": nested_subset(
            source_data["train"], count, progress["config"]["seed"], {e["group"] for e in required}
        ),
    }
    audit = teacher_statistics(data)
    atomic_save(output / "teacher.pt", data)
    for name in ("config.json", "prompts.json"):
        (output / name).write_bytes((parent / name).read_bytes())
    write_json(output / "teacher-audit.json", audit)
    report = {
        "completed": True,
        "config": progress["config"],
        "teacher_audit": audit,
        "preparation_parent": str(parent),
        "prepared_subset": count,
        "full_pool_completed": progress["cursor"]
        == json.loads((parent / "report.json").read_text())["requested"],
        "note": "Sealed available subset only; full preparation may remain incomplete.",
        "test_status": "test labels/features/metrics not generated",
    }
    write_json(output / "report.json", report)
    return report
