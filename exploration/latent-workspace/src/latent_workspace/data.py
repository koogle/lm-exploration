"""Split before windowing; no window crosses document or held-out boundaries."""

import hashlib

import torch


def documents(rows):
    # WikiText's top-level headings identify articles; paragraphs are NOT documents.
    parts = []
    for row in rows:
        line = row["text"].strip()
        if line.startswith("= ") and line.endswith(" =") and not line.startswith("= ="):
            if parts:
                yield "\n".join(parts)
            parts = []
        if line:
            parts.append(line)
    if parts:
        yield "\n".join(parts)


def make_windows(texts, tokenizer, context, block, limit, excluded=None):
    windows, hashes = [], set()
    excluded = excluded or set()
    for text in texts:
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in excluded or digest in hashes:
            continue
        hashes.add(digest)
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Disjoint windows within each article, no padding and no future prefix.
        length = context + block
        for start in range(0, len(ids) - length + 1, length):
            windows.append(ids[start : start + length])
            if len(windows) >= limit:
                return torch.tensor(windows, dtype=torch.long), hashes
    if not windows:
        raise ValueError("no full windows: reduce context or use more data")
    return torch.tensor(windows, dtype=torch.long), hashes


def load_data(cfg, tokenizer):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(cfg.dataset, revision=cfg.dataset_revision).sha

    def rows(split):
        return load_dataset(
            cfg.dataset, cfg.dataset_config, split=split, revision=revision, streaming=True
        )

    valid, valid_hashes = make_windows(
        documents(rows("validation")),
        tokenizer,
        cfg.context_length,
        cfg.block_size,
        cfg.eval_examples,
    )
    train, train_hashes = make_windows(
        documents(rows("train")),
        tokenizer,
        cfg.context_length,
        cfg.block_size,
        cfg.train_examples,
        valid_hashes,
    )
    return (
        train,
        valid,
        {
            "dataset_revision": revision,
            "train_document_hashes": sorted(train_hashes),
            "validation_document_hashes": sorted(valid_hashes),
            "train_examples": len(train),
            "validation_examples": len(valid),
            "train_window_tokens": train.numel(),
            "validation_window_tokens": valid.numel(),
        },
    )
