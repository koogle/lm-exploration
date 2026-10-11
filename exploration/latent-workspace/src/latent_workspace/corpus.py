"""Balanced prose/code/translation corpus with frozen train/validation/test splits."""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice

import torch

from .data import documents

DOMAINS = ("prose", "code", "translation")
SPLITS = ("train", "validation", "test")
SOURCES = {
    "prose": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    "code": ("codeparrot/codeparrot-clean", None),
    "translation": ("wmt/wmt14", "de-en"),
}


def digest(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def repository_split(repository):
    value = int(digest("latent-workspace-v1:" + repository), 16) % 100
    return "test" if value < 10 else "validation" if value < 20 else "train"


@dataclass
class Record:
    domain: str
    split: str
    group: str
    content: str
    source: str = ""
    cached_windows: list | None = None

    @property
    def fingerprint(self):
        # Source-only deduplication prevents translation source leakage even if
        # alternative target translations differ across official partitions.
        return digest(self.source if self.domain == "translation" else self.content)


def record_windows(record, tokenizer, context, block):
    if record.cached_windows is not None:
        yield from record.cached_windows
        return
    if record.domain != "translation":
        ids = tokenizer.encode(record.content, add_special_tokens=False)
        for start in range(0, len(ids) - context - block + 1, context + block):
            yield ids[start : start + context + block]
    else:
        prompt = "Translate English to German.\nEnglish: " + record.source + "\nGerman:"
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        target = tokenizer.encode(" " + record.content, add_special_tokens=False)
        # Only target-language tokens receive labels; never window source text
        # as a translation target. -1 marks left padding, masked in the backbone.
        for offset in range(0, len(target) - block + 1, block):
            context_ids = (prefix + target[:offset])[-context:]
            yield (
                [-1] * (context - len(context_ids)) + context_ids + target[offset : offset + block]
            )


def build(records, tokenizer, context, block, limits, excluded_groups=(), excluded_hashes=()):
    """Collect disjoint groups first, then tokenize; exact duplicates are quarantined."""
    excluded_groups, excluded_hashes = set(excluded_groups), set(excluded_hashes)
    pools = defaultdict(list)
    counts = defaultdict(int)
    fingerprints = {}
    duplicate_hashes = set()
    for record in records:
        if record.split == "train" and (
            f"{record.domain}:{record.group}" in excluded_groups
            or record.fingerprint in excluded_hashes
        ):
            continue
        key = (record.split, record.domain)
        if counts[key] >= limits[record.split] * 3:
            continue
        fingerprint = record.fingerprint
        previous = fingerprints.get(fingerprint)
        if previous is not None:
            if previous != key:
                duplicate_hashes.add(fingerprint)
            continue
        fingerprints[fingerprint] = key
        windows = list(islice(record_windows(record, tokenizer, context, block), 4))
        if windows:
            pools[key].append((record, windows))
            counts[key] += len(windows)
    tensors, ranges, groups, hashes = {}, {}, {}, {}
    reserved_windows = set()
    for split in ("test", "validation", "train"):
        rows, ranges[split], groups[split], hashes[split] = [], {}, [], []
        for domain in DOMAINS:
            start = len(rows)
            seen_windows = set()
            for record, windows in pools[(split, domain)]:
                if record.fingerprint in duplicate_hashes:
                    continue
                remaining = limits[split] - (len(rows) - start)
                if remaining <= 0:
                    break
                chosen = []
                for window in windows:
                    key = tuple(window)
                    if key in seen_windows or key in reserved_windows:
                        continue
                    seen_windows.add(key)
                    chosen.append(window)
                    if len(chosen) >= remaining:
                        break
                rows.extend(chosen)
                reserved_windows.update(map(tuple, chosen))
                groups[split].extend([f"{domain}:{record.group}"] * len(chosen))
                hashes[split].extend([record.fingerprint] * len(chosen))
            end = len(rows)
            if end - start < limits[split]:
                raise ValueError(
                    f"insufficient disjoint {split}/{domain} windows: {end - start}/{limits[split]}"
                )
            ranges[split][domain] = [start, end]
        tensors[split] = torch.tensor(rows, dtype=torch.long)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(groups[a]) & set(groups[b]) or set(hashes[a]) & set(hashes[b]):
            raise ValueError(f"split leakage: {a}/{b}")
    metadata = {
        "domain_ranges": ranges,
        "window_groups": groups,
        "window_hashes": hashes,
        "split_policy": "mixed-v1",
        "quarantined_duplicate_hashes": sorted(duplicate_hashes),
    }
    return tensors, metadata


def prepare(cfg, tokenizer, frozen_metadata=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revisions = {
        domain: frozen_metadata["sources"][domain]["revision"]
        if frozen_metadata
        else HfApi().dataset_info(repo).sha
        for domain, (repo, _) in SOURCES.items()
    }
    limits = {
        "train": cfg.train_examples // 3,
        "validation": cfg.eval_examples // 3,
        "test": cfg.test_examples // 3,
    }

    # Oversample before exact-duplicate quarantine; quotas are checked afterwards.
    def records():
        for domain in DOMAINS:
            repo, subset = SOURCES[domain]
            splits = ("train",) if domain == "code" else ("test", "validation", "train")
            for split in splits:
                print(f"Collecting {domain}/{split} windows", flush=True)
                rows = load_dataset(
                    repo, subset, split=split, revision=revisions[domain], streaming=True
                )
                if domain == "prose":
                    rows = ({"text": text} for text in documents(rows))
                counts = defaultdict(int)
                for index, row in enumerate(rows):
                    if index >= 100000:
                        break
                    if domain == "code":
                        group = row["repo_name"]
                        partition = repository_split(group)
                        record = Record(domain, partition, group, row["content"])
                    elif domain == "translation":
                        pair = row["translation"]
                        record = Record(domain, split, digest(pair["en"]), pair["de"], pair["en"])
                    else:
                        record = Record(domain, split, digest(row["text"]), row["text"])
                    key = record.split
                    quota = limits[key] * 3
                    if counts[key] < quota:
                        record.cached_windows = list(
                            islice(
                                record_windows(
                                    record, tokenizer, cfg.context_length, cfg.block_size
                                ),
                                4,
                            )
                        )
                        if record.cached_windows:
                            yield record
                            counts[key] += len(record.cached_windows)
                    required = SPLITS if domain == "code" else (split,)
                    if all(counts[x] >= limits[x] * 3 for x in required):
                        break
                print(f"Collected {domain}/{split}: {dict(counts)} candidate windows", flush=True)

    excluded = frozen_metadata or {"window_groups": {}, "window_hashes": {}}
    tensors, metadata = build(
        records(),
        tokenizer,
        cfg.context_length,
        cfg.block_size,
        limits,
        [g for split in ("validation", "test") for g in excluded["window_groups"].get(split, [])],
        [h for split in ("validation", "test") for h in excluded["window_hashes"].get(split, [])],
    )
    metadata["sources"] = {
        domain: {
            "repository": SOURCES[domain][0],
            "config": SOURCES[domain][1],
            "revision": revision,
        }
        for domain, revision in revisions.items()
    }
    metadata["dataset_revision"] = digest(json.dumps(metadata["sources"], sort_keys=True))
    return tensors, metadata


def extend_frozen_corpus(base, candidate, target_size):
    """Keep held-out tensors fixed; retain clean old rows and add distinct windows."""
    import copy

    old, old_meta = base["tensors"], base["metadata"]
    fresh, fresh_meta = candidate
    if old_meta["sources"] != fresh_meta["sources"]:
        raise ValueError("corpus expansion must retain source revisions")
    metadata = copy.deepcopy(old_meta)
    metadata["window_groups"]["train"] = []
    metadata["window_hashes"]["train"] = []
    rows, priority = [], {}
    excluded_groups = set(
        old_meta["window_groups"]["validation"] + old_meta["window_groups"]["test"]
    )
    excluded_hashes = set(
        old_meta["window_hashes"]["validation"] + old_meta["window_hashes"]["test"]
    )
    excluded_windows = {
        tuple(row) for split in ("validation", "test") for row in old[split].tolist()
    }
    removed_duplicates, removed_overlaps = 0, 0
    for domain in DOMAINS:
        start = len(rows)
        a, b = old_meta["domain_ranges"]["train"][domain]
        seen, priority[domain] = {}, {}
        for tensors, meta, bounds in (
            (old, old_meta, (a, b)),
            (fresh, fresh_meta, fresh_meta["domain_ranges"]["train"][domain]),
        ):
            for index in range(*bounds):
                if len(rows) - start >= target_size // 3:
                    break
                group = meta["window_groups"]["train"][index]
                fingerprint = meta["window_hashes"]["train"][index]
                if group in excluded_groups or fingerprint in excluded_hashes:
                    raise ValueError("expansion leaks held-out records")
                row = tensors["train"][index]
                key = tuple(row.tolist())
                if key in excluded_windows:
                    removed_overlaps += tensors is old
                    continue
                if key in seen:
                    if tensors is old:
                        priority[domain][index] = seen[key]
                        removed_duplicates += 1
                    continue
                seen[key] = len(rows)
                if tensors is old:
                    priority[domain][index] = len(rows)
                rows.append(row)
                metadata["window_groups"]["train"].append(group)
                metadata["window_hashes"]["train"].append(fingerprint)
        if len(rows) - start != target_size // 3:
            raise ValueError(
                f"insufficient unique expansion rows for {domain}: {len(rows) - start}/{target_size // 3}"
            )
        metadata["domain_ranges"]["train"][domain] = [start, len(rows)]
    metadata["parent_training_cleanup"] = {
        "duplicate_windows_removed": removed_duplicates,
        "held_out_overlapping_windows_removed": removed_overlaps,
        "note": "Held-out tensors unchanged; historical training subset may change to remove leakage/duplicates.",
    }
    return {**old, "train": torch.stack(rows)}, metadata, priority


def audit_windows(data):
    """Fail before training if any exact window duplicates or split overlaps remain."""
    sets = {split: set(map(tuple, rows.tolist())) for split, rows in data.items()}
    counts = {
        split: {"total": len(data[split]), "distinct": len(values)}
        for split, values in sets.items()
    }
    overlaps = {
        f"{a}/{b}": len(sets[a] & sets[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(c["total"] != c["distinct"] for c in counts.values()) or any(overlaps.values()):
        raise ValueError(
            "window audit failed: duplicates or held-out overlap; clean/expand the parent corpus first"
        )
    return {"counts": counts, "overlaps": overlaps}
