"""Convert accepted teacher trajectories into flat NumPy shards.

Each shard holds the concatenated output token IDs (native Qwen IDs, ending in the
teacher's stop token), document offsets, the teacher's top-K IDs and log
probabilities per position, and the native vocabulary counts. The generation
prompt is dropped: the student learns the exercise text itself.
"""

import gzip
import json
from pathlib import Path

import numpy as np
from common import QWEN_VOCAB, TOPK, atomic_json, file_sha256


def read_chunk(path):
    """Yield accepted records as (token_ids, topk_ids, topk_logprobs, record)."""
    with gzip.open(path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            if not record["accepted"] or record["finish_reason"] != "stop":
                continue
            tokens = record["token_ids"]
            targets = record["targets"]
            if not (
                len(tokens) == len(targets["topk_ids"]) == len(targets["topk_logprobs"])
            ):
                raise ValueError(f"misaligned targets in {record['id']}")
            yield tokens, targets["topk_ids"], targets["topk_logprobs"], record


def write_shard(chunks, out_dir, keep_records=False):
    """Write one shard from local chunk files; returns its summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens, lengths, top_ids, top_lp, records = [], [], [], [], []
    for chunk in chunks:
        for ids, k_ids, k_lp, record in read_chunk(chunk):
            tokens.append(np.asarray(ids, dtype=np.int32))
            lengths.append(len(ids))
            top_ids.append(np.asarray(k_ids, dtype=np.int32)[:, :TOPK])
            top_lp.append(
                np.asarray(k_lp, dtype=np.float32)[:, :TOPK].astype(np.float16)
            )
            if keep_records:
                text = json.loads(record["text"])
                records.append({"id": record["id"], "prompt": record["prompt"], **text})
    if not tokens:
        raise ValueError("shard has no accepted records")
    tokens = np.concatenate(tokens)
    arrays = {
        "tokens": tokens,
        "offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "topk_ids": np.concatenate(top_ids),
        "topk_logprobs": np.concatenate(top_lp),
        "counts": np.bincount(tokens, minlength=QWEN_VOCAB).astype(np.int64),
    }
    offsets = arrays["offsets"]
    summary = {
        "chunks": len(chunks),
        "documents": int(offsets.size - 1),
        "tokens": int(tokens.size),
    }
    for name, array in arrays.items():
        np.save(out_dir / f"{name}.npy", array)
        summary[f"{name}_sha256"] = file_sha256(out_dir / f"{name}.npy")
    if keep_records:
        (out_dir / "records.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
    atomic_json(out_dir / "summary.json", summary)
    return summary


def build_vocab(counts, size):
    """Trimmed vocabulary: PAD, BOS, UNK, then the most frequent native IDs.

    Returns the native-to-student lookup table (unknown IDs map to UNK) and the
    kept native IDs in student order.
    """
    specials = 3
    order = np.argsort(-counts, kind="stable")
    kept = order[: size - specials]
    kept = kept[counts[kept] > 0]
    table = np.full(QWEN_VOCAB, 2, dtype=np.int32)
    table[kept] = np.arange(specials, specials + kept.size, dtype=np.int32)
    return table, kept


def byte_lengths(tokenizer_json):
    """UTF-8 byte length of each native token; special tokens count as zero bytes."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1)]
    printable += range(ord("®"), ord("ÿ") + 1)
    decoder, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            decoder[chr(byte)] = byte
        else:
            decoder[chr(256 + extra)] = byte
            extra += 1
    lengths = np.zeros(QWEN_VOCAB, dtype=np.int64)
    for token, index in tokenizer_json["model"]["vocab"].items():
        lengths[index] = sum(1 for char in token if char in decoder)
    for added in tokenizer_json.get("added_tokens", []):
        lengths[added["id"]] = 0
    return lengths
