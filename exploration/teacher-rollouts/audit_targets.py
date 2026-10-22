"""Audit a saved gzip JSONL shard without loading a teacher or GPU."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def audit(path):
    records = tokens = accepted = 0
    identities = set()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in identities:
                raise ValueError("Duplicate record identity in shard")
            identities.add(row["id"])
            ids, targets = row["token_ids"], row["targets"]
            required = ["topk_ids", "topk_logprobs", "selected_logprobs", "tail_mass"]
            if not ids or any(len(targets[k]) != len(ids) for k in required):
                raise ValueError("Target/token length mismatch")
            if row["accepted"] and row["finish_reason"] != "stop":
                raise ValueError("Truncated output accepted")
            for i, token in enumerate(ids):
                top_ids, logp = targets["topk_ids"][i], targets["topk_logprobs"][i]
                selected, tail = (
                    targets["selected_logprobs"][i],
                    targets["tail_mass"][i],
                )
                if len(top_ids) != 64 or len(set(top_ids)) != 64 or len(logp) != 64:
                    raise ValueError("Invalid top-64 shape")
                if not all(
                    isinstance(t, int) and 0 <= t < 248320 for t in [token, *top_ids]
                ):
                    raise ValueError("Invalid native teacher token ID")
                if not all(math.isfinite(p) and p <= 2e-5 for p in [selected, *logp]):
                    raise ValueError("Invalid log probability")
                mass = sum(math.exp(p) for p in logp)
                if not 0 <= tail <= 1 or abs(mass + tail - 1) > 2e-5:
                    raise ValueError("Probability mass not conserved")
                if token in top_ids:
                    if abs(selected - logp[top_ids.index(token)]) > 2e-5:
                        raise ValueError("Selected-token probability misaligned")
                elif math.exp(selected) > tail + 2e-5:
                    raise ValueError("Selected token exceeds residual mass")
            records += 1
            tokens += len(ids)
            accepted += row["accepted"]
    return {
        "records": records,
        "tokens": tokens,
        "accepted_records": accepted,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "passed": True,
        "scope": "Storage/probability integrity, not semantic correctness",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(audit(args.path), indent=2))
