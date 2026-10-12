"""Prototype extra training-only article prefixes; not wired into GPU preparation."""

import hashlib

from .corpus import repository_split


def training_prose_prefixes(
    tokens,
    source_hash,
    *,
    context,
    seed,
    existing_offsets=(),
    max_prefixes=4,
    excluded_groups=(),
    reserve_tokens=8,
):
    """Yield nonoverlapping prompts while retaining the article's split identity.

    `max_prefixes` includes existing prompts; extra example IDs never redefine the
    split group. Historical held-out articles must be supplied in excluded_groups.
    Registry validation remains necessary for cross-source and prompt deduplication.
    """
    if min(context, max_prefixes, reserve_tokens) < 1:
        raise ValueError("positive context, prefix limit and trailing-token reserve required")
    offsets = list(existing_offsets)
    if any(type(offset) is not int or offset < 0 for offset in offsets):
        raise ValueError("existing offsets must be nonnegative integers")
    offsets = sorted(set(offsets))
    group = f"prose:{source_hash}"
    if group in excluded_groups or repository_split(group) != "train":
        return
    candidates = range(0, len(tokens) - context - reserve_tokens + 1, 2 * context)
    candidates = sorted(
        candidates,
        key=lambda offset: hashlib.sha256(f"{seed}:{group}:{offset}".encode()).digest(),
    )
    for offset in candidates:
        if len(offsets) >= max_prefixes:
            return
        if any(abs(offset - previous) < context for previous in offsets):
            continue
        offsets.append(offset)
        yield {
            "domain": "prose",
            "split": "train",
            "group": f"{group}:prefix:{offset}",
            "source_group": group,
            "source_hash": source_hash,
            "source_token_offset": offset,
            "prompt_ids": list(tokens[offset : offset + context]),
        }
