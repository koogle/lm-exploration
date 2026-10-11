"""Diverse, disjoint target-response pool for matched DFlash data-size experiments."""

import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from .corpus import DOMAINS, SOURCES, SPLITS, digest, repository_split
from .data import documents
from .dflash_data import PromptRegistry
from .train import write_json


def buffer_shuffle(rows, seed, size=4096):
    """Deterministic bounded-memory shuffle; preserves documents before mixing."""
    rng = random.Random(seed)
    buffer = []
    for row in rows:
        if len(buffer) < size:
            buffer.append(row)
        else:
            i = rng.randrange(size)
            yield buffer[i]
            buffer[i] = row
    rng.shuffle(buffer)
    yield from buffer


def continuation_slice(ids, width, group, seed):
    if len(ids) < 32:
        return None
    rng = random.Random(f"{seed}:{group}")
    start = rng.randrange(max(1, len(ids) - width - 8 + 1))
    return ids[start : start + width], start


def nested_subset(examples, count, seed):
    if count <= 0 or count % 3:
        raise ValueError("subset must be a positive multiple of three")
    result = []
    for domain in DOMAINS:
        rows = sorted(
            (e for e in examples if e["domain"] == domain),
            key=lambda e: hashlib.sha256(f"{seed}:{e['group']}".encode()).hexdigest(),
        )
        if len(rows) < count // 3:
            raise ValueError(f"not enough usable {domain} responses for subset {count}")
        result.extend(rows[: count // 3])
    return result


def diverse_manifest(cfg, tokenizer, output, deadline, historical, prior_manifest):
    from datasets import load_dataset

    old_groups = historical["window_groups"]
    old_all = {g for values in old_groups.values() for g in values}
    old_heldout = {g for split in ("validation", "test") for g in old_groups[split]}
    registry = PromptRegistry()
    for record in prior_manifest["records"]:
        if not registry.add(dict(record)):
            raise ValueError("prior manifest is not internally disjoint")
    limits = dict(
        zip(SPLITS, (cfg.train_prompts // 3, cfg.validation_prompts // 3, cfg.test_prompts // 3))
    )
    counts, rejected = defaultdict(int), defaultdict(int)
    records = []
    for domain_index, domain in enumerate(DOMAINS):
        repo, subset = SOURCES[domain]
        revision = historical["sources"][domain]["revision"]
        rows = load_dataset(repo, subset, split="train", revision=revision, streaming=True)
        if domain == "prose":
            rows = ({"text": text} for text in documents(rows))
        rows = buffer_shuffle(rows, cfg.seed + domain_index)
        for row in rows:
            if time.monotonic() > deadline:
                raise TimeoutError("diverse prompt preparation deadline")
            text = (
                row["translation"]["en"]
                if domain == "translation"
                else row["content"]
                if domain == "code"
                else row["text"]
            )
            source_hash = digest(text)
            group = f"{domain}:" + (row["repo_name"] if domain == "code" else source_hash)
            split = repository_split(row["repo_name"] if domain == "code" else group)
            if counts[domain, split] >= limits[split]:
                continue
            if group in (old_heldout if split == "train" else old_all):
                rejected["historical_group"] += 1
                continue
            if domain == "translation":
                ids = tokenizer.encode(
                    "Translate English to German.\nEnglish: " + text + "\nGerman:",
                    add_special_tokens=False,
                )
                if len(ids) > cfg.prompt_tokens:
                    rejected["translation_prompt_too_long"] += 1
                    continue
                offset = 0
            else:
                # Tokenize only bounded source text; the full source hash remains the group identity.
                ids = tokenizer.encode(text[:100000], add_special_tokens=False)
                piece = continuation_slice(ids, cfg.prompt_tokens, group, cfg.seed)
                if piece is None:
                    continue
                ids, offset = piece
            record = {
                "domain": domain,
                "split": split,
                "group": group,
                "source_hash": source_hash,
                "prompt_ids": ids,
                "source_token_offset": offset,
            }
            if registry.add(record):
                records.append(record)
                counts[domain, split] += 1
            if all(counts[domain, s] == limits[s] for s in SPLITS):
                break
        print(f"Diverse prompts {domain}: {dict(counts)}", flush=True)
    if any(counts[d, s] != limits[s] for d in DOMAINS for s in SPLITS):
        raise ValueError(f"insufficient diverse prompts: {dict(counts)}")
    manifest = {
        "records": records,
        "sources": historical["sources"],
        "seed": cfg.seed,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "sampling": "4096-item seeded buffer shuffle after prose document assembly; deterministic varied prose/code token offsets; one prompt per group; first 100000 source characters maximum",
        "split_policy": "only official source train pools; deterministic 80/10/10 group hash; code by repository, prose by whole document hash, translation by English source sentence hash",
        "quarantine": "prior DFlash manifest registered before sampling; new held-out groups exclude all historical groups; new training excludes historical held-out groups",
        "near_duplicate_policy": "cross-split prompt token-5-gram Jaccard >=0.8; not a comprehensive document/translation-family audit",
        "rejected": dict(rejected),
        "registry_rejected": dict(registry.rejected),
        "test_status": "test prompts only; no teacher responses or metrics",
    }
    write_json(Path(output) / "prompts.json", manifest)
    return manifest


def teacher_statistics(data, block_size=8):
    by_split = {}
    for split, rows in data.items():
        by_split[split] = {}
        for domain in DOMAINS:
            responses = [
                e["tokens"][e["prompt_length"] :].tolist() for e in rows if e["domain"] == domain
            ]
            by_split[split][domain] = {
                "prompts": len(responses),
                "response_tokens": sum(map(len, responses)),
                "eligible_anchors": sum(max(0, len(r) - block_size + 1) for r in responses),
                "mean_unique_response_4gram_fraction": sum(
                    len({tuple(r[i : i + 4]) for i in range(len(r) - 3)}) / max(1, len(r) - 3)
                    for r in responses
                )
                / max(1, len(responses)),
                "response_length_min": min(map(len, responses), default=0),
                "response_length_max": max(map(len, responses), default=0),
            }
    return {"by_split_domain": by_split}


def generate_batch(target, records, response_tokens, eos_id):
    """Left pad prompts for generation; remove padding and stop at the first EOS."""
    import torch

    width = max(len(r["prompt_ids"]) for r in records)
    ids = torch.full((len(records), width), eos_id, dtype=torch.long, device=target.device)
    mask = torch.zeros_like(ids)
    for i, record in enumerate(records):
        length = len(record["prompt_ids"])
        ids[i, -length:] = torch.tensor(record["prompt_ids"], device=target.device)
        mask[i, -length:] = 1
    with torch.no_grad():
        generated = (
            target.generate(
                ids,
                attention_mask=mask,
                max_new_tokens=response_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                pad_token_id=eos_id,
                eos_token_id=eos_id,
            )[:, width:]
            .cpu()
            .tolist()
        )
    sequences = []
    for record, response in zip(records, generated):
        if eos_id in response:
            response = response[: response.index(eos_id) + 1]
        sequences.append(record["prompt_ids"] + response)
    return sequences


def build_batched_teacher(cfg, target, tokenizer, manifest, output, deadline):
    import torch

    from .dflash import features
    from .dflash_data import token_hash
    from .train import atomic_save

    examples = {"train": [], "validation": []}
    rows = sorted(
        (r for r in manifest["records"] if r["split"] != "test"),
        key=lambda r: r["split"] != "validation",
    )
    seen = set()
    short, duplicates = 0, 0
    started = time.monotonic()
    for offset in range(0, len(rows), cfg.teacher_batch_size):
        if time.monotonic() > deadline:
            atomic_save(Path(output) / "teacher-partial.pt", examples)
            raise TimeoutError("batched teacher deadline; partial cache retained")
        records = rows[offset : offset + cfg.teacher_batch_size]
        sequences = generate_batch(target, records, cfg.response_tokens, tokenizer.eos_token_id)
        width = max(map(len, sequences))
        tokens = torch.full(
            (len(records), width), tokenizer.eos_token_id, dtype=torch.long, device=target.device
        )
        mask = torch.zeros_like(tokens)
        for i, sequence in enumerate(sequences):
            tokens[i, : len(sequence)] = torch.tensor(sequence, device=target.device)
            mask[i, : len(sequence)] = 1
        with torch.no_grad():
            result = target.model(
                tokens, attention_mask=mask, use_cache=False, output_hidden_states=True
            )
            memory = features(result, cfg.layers).cpu()
        for i, (record, sequence) in enumerate(zip(records, sequences)):
            response = sequence[len(record["prompt_ids"]) :]
            if len(response) < cfg.block_size:
                short += 1
                continue
            response_hash = token_hash(response)
            if response_hash in seen:
                duplicates += 1
                continue
            seen.add(response_hash)
            examples[record["split"]].append(
                {
                    **record,
                    "tokens": torch.tensor(sequence),
                    "prompt_length": len(record["prompt_ids"]),
                    "memory": memory[i, : len(sequence)].clone(),
                    "response_sha256": response_hash,
                }
            )
        if offset % (cfg.teacher_batch_size * 8) == 0:
            print(
                f"Batched teacher {min(offset + len(records), len(rows))}/{len(rows)} ({time.monotonic() - started:.1f}s)",
                flush=True,
            )
    audit = teacher_statistics(examples, cfg.block_size)
    audit.update(
        short_responses=short,
        duplicate_responses=duplicates,
        teacher_batch_size=cfg.teacher_batch_size,
        target_arithmetic=str(target.dtype),
        note="Batched greedy target generation; BF16 batch-shape argmax can differ from single-prompt arithmetic. Both data-size fits use this identical cache.",
    )
    if any(audit["by_split_domain"][s][d]["prompts"] < 2 for s in examples for d in DOMAINS):
        raise ValueError("not enough teacher examples per split/domain")
    atomic_save(Path(output) / "teacher.pt", examples)
    write_json(Path(output) / "teacher-audit.json", audit)
    return examples, audit


def prepare_pool(cfg, output, on_checkpoint=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .train import seed_all

    cfg.validate()
    if cfg.block_size != 8:
        raise ValueError("current data-scale audit is for seven-new-token blocks")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + cfg.max_seconds - 120
    seed_all(cfg.seed)
    write_json(output / "config.json", cfg.to_dict())
    historical = torch.load(cfg.historical_corpus, map_location="cpu", weights_only=True)[
        "metadata"
    ]
    prior = json.loads(Path("/runs/dflash-pilot-02/prompts.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    if cfg.prompt_manifest:
        manifest = json.loads(Path(cfg.prompt_manifest).read_text())
        source_tokenizer = AutoTokenizer.from_pretrained(
            manifest["model"], revision=manifest["model_revision"]
        )
        source_hash = hashlib.sha256(Path(cfg.prompt_manifest).read_bytes()).hexdigest()
        counts = defaultdict(int)
        limits = {
            "train": cfg.train_prompts // 3,
            "validation": cfg.validation_prompts // 3,
            "test": cfg.test_prompts // 3,
        }
        selected = []
        for record in manifest["records"]:
            key = (record["domain"], record["split"])
            if counts[key] < limits[record["split"]]:
                selected.append(record)
                counts[key] += 1
        manifest["records"] = selected
        for record in manifest["records"]:
            text = source_tokenizer.decode(record["prompt_ids"], skip_special_tokens=False)
            if cfg.teacher_prompt_style == "chat_no_thinking":
                if record["domain"] == "prose":
                    text = (
                        "Continue the following prose. Output only the continuation, without repeating the supplied text or adding commentary.\n\n"
                        + text
                    )
                elif record["domain"] == "code":
                    text = (
                        "Continue the following code. Output only the continuation, without repeating the supplied code, explanation, or Markdown fences.\n\n"
                        + text
                    )
                else:
                    text = "Output only the German translation, without explanation.\n" + text
                record["prompt_ids"] = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
            else:
                record["prompt_ids"] = tokenizer.encode(text, add_special_tokens=False)
        manifest.update(
            model=cfg.model,
            model_revision=cfg.model_revision,
            parent_manifest_sha256=source_hash,
            adaptation="same decoded source prompts, retokenized without truncation",
            teacher_prompt_style=cfg.teacher_prompt_style,
        )
        # Retokenization changes token identities; repeat the cross-split audit.
        registry = PromptRegistry()
        for record in manifest["records"]:
            if not registry.add(record):
                raise ValueError("retokenized manifest failed deduplication")
        write_json(output / "prompts.json", manifest)
    else:
        manifest = diverse_manifest(cfg, tokenizer, output, deadline, historical, prior)
    if on_checkpoint:
        on_checkpoint()
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model, revision=cfg.model_revision, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    data, audit = build_batched_teacher(cfg, target, tokenizer, manifest, output, deadline)
    source = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        source.update(file.name.encode() + file.read_bytes())
    report = {
        "config": cfg.to_dict(),
        "teacher_audit": audit,
        "validation_samples": [
            {
                "domain": d,
                "prompt": tokenizer.decode(e["tokens"][: e["prompt_length"]]),
                "response": tokenizer.decode(e["tokens"][e["prompt_length"] :]),
            }
            for d in DOMAINS
            for e in [next(x for x in data["validation"] if x["domain"] == d)]
        ],
        "elapsed_seconds": time.monotonic() - started,
        "completed": True,
        "source_sha256": source.hexdigest(),
        "test_status": manifest["test_status"],
    }
    write_json(output / "report.json", report)
    return report
