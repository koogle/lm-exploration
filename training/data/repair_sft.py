"""Strict JSONL repair examples; no supplied Python is executed during preparation."""

import hashlib
import json
from pathlib import Path

from training_data.subword import BOS_TOKEN_ID, EOS_TOKEN_ID, LOSS_IGNORE_INDEX

FORMAT = "repair-sft-v1"


def prompt(row):
    return (
        "Repair the Python code to satisfy the task. Return only the complete corrected code.\n"
        f"Task: {row['instruction']}\nBroken code:\n{row['broken']}\nCorrected code:\n"
    )


def encode(row, tokenizer, context):
    prefix = [BOS_TOKEN_ID] + tokenizer.encode(prompt(row), add_special_tokens=False).ids
    answer = tokenizer.encode(row["fixed"], add_special_tokens=False).ids + [EOS_TOKEN_ID]
    if len(prefix) + len(answer) - 1 > context:
        raise ValueError(f"Example {row['id']} exceeds context; refusing to truncate repair/EOS")
    tokens = prefix + answer
    labels = [LOSS_IGNORE_INDEX] * len(prefix) + answer
    return {"id": row["id"], "x": tokens[:-1], "y": labels[1:]}


def prepare(path, tokenizer, context):
    raw = Path(path).read_bytes()
    splits = {"train": [], "validation": []}
    ids, groups, contents = set(), {}, {}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "group", "split", "instruction", "broken", "fixed", "source", "license"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing nonempty string: {key}")
        split = row["split"]
        if split not in splits or row["id"] in ids:
            raise ValueError("Invalid split or duplicate ID")
        ids.add(row["id"])
        if row["broken"].strip() == row["fixed"].strip():
            raise ValueError("Repair must change code")
        # Treat group as repository/change-family identity, not individual example ID.
        group = row["group"]
        if groups.setdefault(group, split) != split:
            raise ValueError("Group crosses train/validation")
        for field in ("broken", "fixed"):
            fingerprint = " ".join(row[field].split())
            if contents.setdefault(fingerprint, split) != split:
                raise ValueError("Code duplicate crosses train/validation")
        splits[split].append(encode(row, tokenizer, context))
    if not all(splits.values()):
        raise ValueError("Both train and validation examples are required")
    return splits, hashlib.sha256(raw).hexdigest()
