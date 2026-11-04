"""Describe repetition inside recorded truncated replies, including unparsed calls."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    censor = (
        json.loads((root / "censored.json").read_text())
        if (root / "censored.json").exists()
        else None
    )
    rows = []
    for path in sorted((root / "live").glob("*/*/raw.jsonl")):
        variant, batch = path.parent.parent.name, path.parent.name
        for raw in map(json.loads, path.read_text().splitlines()):
            if (
                censor
                and raw["id"] == censor["id"]
                and int(batch.split("-")[0][4:]) == censor["seed"]
            ):
                continue
            if raw["finish_reason"] != "length":
                continue
            text = raw["text"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            grams = [tuple(tokens[i : i + 8]) for i in range(max(0, len(tokens) - 7))]
            counts = Counter(grams)
            rows.append(
                {
                    "variant": variant,
                    "batch": batch,
                    "id": raw["id"],
                    "turn": raw["turn"],
                    "tokens": raw["tokens"],
                    "retokenized_tokens": len(tokens),
                    "repeat_8gram_fraction": 1 - len(counts) / max(1, len(grams)),
                    "max_8gram_count": max(counts.values(), default=0),
                    "grep_v_count": text.count("grep -v"),
                    "possible_patch_text": bool(
                        re.search(r"write_text|write_bytes|apply_patch|sed -i|\.write\(", text)
                    ),
                    "most_repeated_span": tokenizer.decode(max(counts, key=counts.get))
                    if counts
                    else "",
                }
            )
    summary = {}
    for variant in sorted({r["variant"] for r in rows}):
        selected = [r for r in rows if r["variant"] == variant]
        summary[variant] = {
            "truncated_generations": len(selected),
            "half_or_more_8grams_repeated": sum(
                r["repeat_8gram_fraction"] >= 0.5 for r in selected
            ),
            "possible_patch_text": sum(r["possible_patch_text"] for r in selected),
            "repeat_8gram_quantiles": np.quantile(
                [r["repeat_8gram_fraction"] for r in selected], [0, 0.5, 0.9, 1]
            ).tolist(),
        }
    (root / "truncation-analysis.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
