"""Read-only audit of real document-local training windows and source repetition."""

import argparse
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
from sampling.metrics import repetition
from training_data.raw_token_stream import RawTokenMixture
from training_data.subword import load_frozen


def audit(manifest_path):
    data = RawTokenMixture(manifest_path, context=8192, allow_unreviewed=True)
    tok = load_frozen()
    checks = {"windows": 0, "multiwindow_continuations": 0, "eos_targets": 0, "padded_windows": 0}

    def check(block):
        source = block["source"]
        cursor = data.cursors[source]
        raw, expected = data.db.execute(
            "SELECT text,loss_tokens FROM documents WHERE hash=?", (cursor["hash"],)
        ).fetchone()
        text = zlib.decompress(raw).decode()
        ids = tok.encode(text, add_special_tokens=False).ids
        assert len(ids) + 1 == expected
        tokens = [1, *ids, 2]
        offset = cursor["offset"] - 8192
        n = block["loss_tokens"]
        assert block["input_ids"][:n] == tokens[offset : offset + n]
        assert block["labels"][:n] == tokens[offset + 1 : offset + n + 1]
        assert block["input_ids"][n:] == [0] * (8192 - n)
        assert block["labels"][n:] == [-100] * (8192 - n)
        assert all(a == b for a, b in zip(block["input_ids"][1:n], block["labels"][: n - 1]))
        assert 2 not in block["input_ids"][:n]
        checks["windows"] += 1
        checks["multiwindow_continuations"] += offset > 0
        checks["eos_targets"] += block["labels"][n - 1] == 2
        checks["padded_windows"] += n < 8192

    for _ in range(256):
        check(data.next_block())
    state = data.state_dict()
    x, y = data.next_batch(32, "cpu")
    data.load_state_dict(state)
    rx, ry = data.next_batch(32, "cpu")
    import torch

    assert torch.equal(x, rx) and torch.equal(y, ry)
    source_rows = {}
    for name in data.sources:
        data.cursors[name] = {"pass": 0, "hash": "", "offset": 0}
        iterator = data.blocks(name)
        for _ in range(2):
            check(next(iterator))
        values = []
        examples = []
        rows = data.db.execute(
            'SELECT hash,text FROM documents WHERE source=? AND split="train" AND conflict=0 ORDER BY hash LIMIT 64',
            (name,),
        )
        for key, raw in rows:
            text = zlib.decompress(raw).decode()
            ids = tok.encode(text, add_special_tokens=False).ids
            value = repetition(ids, 4)
            values.append(value)
            examples.append(
                {
                    "hash": key,
                    "tokens": len(ids),
                    "repeated_4gram_fraction": value,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
        source_rows[name] = {
            "sampled_documents": len(values),
            "mean_repeated_4gram_fraction": float(np.mean(values)),
            "median_repeated_4gram_fraction": float(np.median(values)),
            "p90_repeated_4gram_fraction": float(np.quantile(values, 0.9)),
            "highest_repetition_documents": sorted(
                examples, key=lambda x: x["repeated_4gram_fraction"], reverse=True
            )[:3],
        }
    counts = {
        split: count
        for split, count in data.db.execute(
            "SELECT split,COUNT(*) FROM documents WHERE conflict=0 GROUP BY split"
        )
    }
    result = {
        "manifest_sha256": data.identity["manifest_sha256"],
        "database_sha256_verified": True,
        "window_checks": checks,
        "all_alignment_boundary_padding_checks_passed": True,
        "next_batch_after_cursor_restore_identical": True,
        "eligible_document_counts": counts,
        "source_repetition_sample": source_rows,
        "notes": [
            "First 64 eligible documents per source in content-hash order; bounded sample, not a full corpus quality audit.",
            "Repeated token 4-grams include legitimate code and mathematical structure; this metric alone is not a quality judgment.",
            "Each checked window belongs to one document; valid targets shift inputs by exactly one token, EOS is supervised, trailing padding labels are ignored.",
        ],
    }
    data.close()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    r = audit(a.manifest)
    Path(a.output).write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "source_repetition_sample"}, indent=2))
