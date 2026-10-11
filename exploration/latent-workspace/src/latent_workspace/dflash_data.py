"""Document-split prompts and target-generated responses for drafting, not human labels."""

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from .corpus import DOMAINS, SOURCES, SPLITS, digest, repository_split
from .data import documents
from .dflash import features
from .train import atomic_save, write_json


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


class PromptRegistry:
    """Exact source/prefix dedup plus cross-split token-5-gram Jaccard quarantine."""

    def __init__(self):
        self.groups = set()
        self.sources = set()
        self.prefixes = set()
        self.shingles = []
        self.index = defaultdict(set)
        self.splits = []
        self.rejected = defaultdict(int)

    def add(self, record):
        ids = record["prompt_ids"]
        key = token_hash(ids)
        if (
            record["group"] in self.groups
            or record["source_hash"] in self.sources
            or key in self.prefixes
        ):
            self.rejected["exact_source_group_or_prompt"] += 1
            return False
        shingles = {tuple(ids[i : i + 5]) for i in range(len(ids) - 4)}
        candidates = set()
        for shingle in shingles:
            candidates.update(self.index[shingle])
        for i in candidates:
            if self.splits[i] == record["split"]:
                continue
            other = self.shingles[i]
            if len(shingles & other) / max(1, len(shingles | other)) >= 0.8:
                self.rejected["cross_split_near_duplicate_prompt"] += 1
                return False
        i = len(self.shingles)
        self.shingles.append(shingles)
        self.splits.append(record["split"])
        for shingle in shingles:
            self.index[shingle].add(i)
        self.groups.add(record["group"])
        self.sources.add(record["source_hash"])
        self.prefixes.add(key)
        record["prompt_sha256"] = key
        return True


def prepare_prompts(cfg, tokenizer, output, deadline):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    registry = PromptRegistry()
    if cfg.prompt_manifest:
        manifest = json.loads(Path(cfg.prompt_manifest).read_text())
        for record in manifest["records"]:
            if record["split"] not in SPLITS or record["domain"] not in DOMAINS:
                raise ValueError("unknown prompt domain/split")
            if not registry.add(record):
                raise ValueError("external manifest fails prompt deduplication")
        write_json(output / "prompts.json", manifest)
        return manifest
    historical = {}
    if cfg.historical_corpus:
        historical = torch.load(cfg.historical_corpus, weights_only=True)["metadata"]
    old_groups = historical.get("window_groups", {})
    old_all = {g for values in old_groups.values() for g in values}
    old_holdout = {g for s in ("validation", "test") for g in old_groups.get(s, [])}
    revisions = {
        domain: historical["sources"][domain]["revision"]
        if historical
        else HfApi().dataset_info(repo).sha
        for domain, (repo, _) in SOURCES.items()
    }
    limits = {
        "train": cfg.train_prompts // 3,
        "validation": cfg.validation_prompts // 3,
        "test": cfg.test_prompts // 3,
    }
    records, counts = [], defaultdict(int)
    for domain in DOMAINS:
        repo, subset = SOURCES[domain]
        for source_split in ("train",) if domain == "code" else ("test", "validation", "train"):
            rows = load_dataset(
                repo, subset, split=source_split, revision=revisions[domain], streaming=True
            )
            if domain == "prose":
                rows = ({"text": text} for text in documents(rows))
            for row in rows:
                if time.monotonic() > deadline:
                    raise TimeoutError("prompt preparation deadline")
                split = repository_split(row["repo_name"]) if domain == "code" else source_split
                if counts[domain, split] >= limits[split]:
                    if all(
                        counts[domain, s] >= limits[s]
                        for s in (SPLITS if domain == "code" else (source_split,))
                    ):
                        break
                    continue
                if domain == "translation":
                    text = row["translation"]["en"]
                    source_hash = digest(text)
                    group = f"{domain}:{source_hash}"
                    text = "Translate English to German.\nEnglish: " + text + "\nGerman:"
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    if len(ids) > cfg.prompt_tokens:
                        continue  # Do not truncate away the translation task or source.
                else:
                    text = row["content"] if domain == "code" else row["text"]
                    source_hash = digest(text)
                    group = f"{domain}:" + (row["repo_name"] if domain == "code" else source_hash)
                    ids = tokenizer.encode(text, add_special_tokens=False)[: cfg.prompt_tokens]
                    if len(ids) < 32:
                        continue
                excluded = old_holdout if split == "train" else old_all
                if group in excluded:
                    continue
                record = {
                    "domain": domain,
                    "split": split,
                    "group": group,
                    "source_hash": source_hash,
                    "prompt_ids": ids,
                }
                if registry.add(record):
                    records.append(record)
                    counts[domain, split] += 1
            print(f"Draft prompts {domain}/{source_split}: {dict(counts)}", flush=True)
    for domain in DOMAINS:
        for split in SPLITS:
            if counts[domain, split] != limits[split]:
                raise ValueError(f"insufficient disjoint prompts: {domain}/{split}")
    manifest = {
        "records": records,
        "sources": {
            d: {"repository": SOURCES[d][0], "config": SOURCES[d][1], "revision": revisions[d]}
            for d in DOMAINS
        },
        "rejected": dict(registry.rejected),
        "split_policy": "official prose/translation splits; code repository hash; one prompt/group",
        "historical_holdout_policy": "new validation/test groups exclude ALL historical groups; training excludes historical held-out groups",
        "near_duplicate_policy": "cross-split token-5-gram Jaccard >=0.8 rejected, at prompt level only",
        "test_status": "prompts reserved; no target responses or test metrics generated",
        "model": cfg.model,
        "model_revision": cfg.model_revision,
    }
    write_json(output / "prompts.json", manifest)
    return manifest


@torch.no_grad()
def build_teacher_cache(cfg, target, tokenizer, manifest, output, deadline, on_checkpoint=None):
    examples = {"train": [], "validation": []}
    stats = {
        "short_responses": 0,
        "duplicate_responses": 0,
        "response_tokens": 0,
        "eligible_anchor_positions": 0,
    }
    response_splits = {}
    device = target.device
    selected = [r for r in manifest["records"] if r["split"] != "test"]
    # Reserve validation first; quarantine exact generated-response collisions.
    selected.sort(key=lambda r: r["split"] != "validation")
    for index, record in enumerate(selected):
        if time.monotonic() > deadline:
            atomic_save(output / "teacher-partial.pt", examples)
            raise TimeoutError("target response preparation deadline; partial data retained")
        prompt = torch.tensor([record["prompt_ids"]], device=device)
        sequence = target.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=cfg.response_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = sequence[0, prompt.shape[1] :].tolist()
        if len(response) < cfg.block_size:
            stats["short_responses"] += 1
            continue
        response_hash = token_hash(response)
        if response_hash in response_splits:
            stats["duplicate_responses"] += 1
            continue
        response_splits[response_hash] = record["split"]
        result = target.model(sequence, use_cache=False, output_hidden_states=True)
        examples[record["split"]].append(
            {
                **record,
                "tokens": sequence[0].cpu(),
                "prompt_length": prompt.shape[1],
                "memory": features(result, cfg.layers)[0].cpu(),
                "response_sha256": response_hash,
            }
        )
        stats["response_tokens"] += len(response)
        stats["eligible_anchor_positions"] += len(response) - cfg.block_size + 1
        if (index + 1) % 32 == 0:
            print(f"Teacher responses {index + 1}/{len(selected)}", flush=True)
    for rows in examples.values():
        if any(sum(x["domain"] == d for x in rows) < 2 for d in DOMAINS):
            raise ValueError("not enough usable teacher responses per domain")
    stats["by_split_domain"] = {}
    for split, rows in examples.items():
        stats["by_split_domain"][split] = {}
        for domain in DOMAINS:
            selected_rows = [e for e in rows if e["domain"] == domain]
            responses = [e["tokens"][e["prompt_length"] :].tolist() for e in selected_rows]
            stats["by_split_domain"][split][domain] = {
                "prompts": len(responses),
                "response_tokens": sum(map(len, responses)),
                "eligible_anchors": sum(len(r) - cfg.block_size + 1 for r in responses),
                "mean_unique_response_4gram_fraction": sum(
                    len({tuple(r[i : i + 4]) for i in range(len(r) - 3)}) / (len(r) - 3)
                    for r in responses
                )
                / len(responses),
            }
    atomic_save(output / "teacher.pt", examples)
    write_json(output / "teacher-audit.json", stats)
    if on_checkpoint:
        on_checkpoint()
    return examples, stats
