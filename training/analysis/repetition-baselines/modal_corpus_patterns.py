"""Read-only bounded corpus-pattern audit; no training or publication.

Run from training/: modal run analysis/repetition-baselines/modal_corpus_patterns.py
CPU only, 10-minute hard timeout, no retries; refuses to start 24 hours after launch.
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("simple-llm-corpus-patterns")
LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tokenizers==0.22.2", "huggingface-hub==0.30.2")
    .env({"PYTHONPATH": "/project"})
    .add_local_dir("analysis/sampling", "/project/sampling")
)
volume = modal.Volume.from_name("lm-exploration-mixture-data")


@app.function(image=image, cpu=4, memory=8192, timeout=600, retries=0, volumes={"/data": volume})
def audit(deadline):
    import hashlib
    import re
    import sqlite3
    import zlib
    from collections import Counter
    from statistics import mean

    from huggingface_hub import hf_hub_download
    from sampling.metrics import repetition
    from tokenizers import Tokenizer

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing the audit")
    base = Path(
        "/data/scholarly-code/scholarly-small-5b-raw-20261017/snapshots/scholarly-modern-10m-5b-20261017"
    )
    manifest = json.loads((base / "manifest.json").read_text())
    path = hf_hub_download(
        "jfrick/scholarly-code-modern-gpt-124m-step2",
        "tokenizer/tokenizer.json",
        revision="38df193c2cc2c5463e4bb20a5c6a4998f58b3af0",
    )
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == manifest["tokenizer_sha256"]
    tokenizer = Tokenizer.from_file(path)
    db = sqlite3.connect(f"{(base / 'raw.sqlite').as_uri()}?mode=ro&immutable=1", uri=True)

    result = {}
    line_document_counts = Counter()
    content_counts = Counter()
    for source in manifest["sources"]:
        rows = []
        for key, parent, raw in db.execute(
            'SELECT hash,parent,text FROM documents WHERE source=? AND split="train" AND conflict=0 ORDER BY hash LIMIT 64',
            (source,),
        ):
            text = zlib.decompress(raw).decode()
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
            lines = [line for line in lines if len(line) >= 20]
            counts = Counter(lines)
            for line in set(lines):
                line_document_counts[hashlib.sha256(line.encode()).hexdigest()] += 1
            content_counts[hashlib.sha256(text.encode()).hexdigest()] += 1
            windows = [ids[i : i + 128] for i in range(0, min(len(ids), 1024) - 127, 128)]
            rows.append(
                {
                    "hash": key,
                    "parent": parent,
                    "tokens": len(ids),
                    "full_document_token_rep4": repetition(ids, 4),
                    "window128_token_rep4": mean(repetition(w, 4) for w in windows)
                    if windows
                    else None,
                    "duplicate_line_fraction": (sum(v - 1 for v in counts.values()) / len(lines))
                    if lines
                    else 0,
                    "eligible_lines": len(lines),
                }
            )
        result[source] = {
            "documents": len(rows),
            "mean_tokens": mean(r["tokens"] for r in rows),
            "mean_full_document_rep4": mean(r["full_document_token_rep4"] for r in rows),
            "mean_window128_rep4": mean(
                r["window128_token_rep4"] for r in rows if r["window128_token_rep4"] is not None
            ),
            "mean_duplicate_line_fraction": mean(r["duplicate_line_fraction"] for r in rows),
            "rows": rows,
        }
    return {
        "manifest_sha256": hashlib.sha256((base / "manifest.json").read_bytes()).hexdigest(),
        "selection": "first64 eligible train documents per source ordered by hash; first8 nonoverlapping complete128token windows/document",
        "sources": result,
        "sampled_exact_duplicate_text_groups": sum(v > 1 for v in content_counts.values()),
        "sampled_cross_document_repeated_lines": sum(v > 1 for v in line_document_counts.values()),
        "largest_line_document_frequencies": line_document_counts.most_common(15),
        "note": "Length-controlled description of sample, not causal attribution or full-corpus near-dedup. Repeated code/math lines may be legitimate. Short documents lack128token window; per-source mean gives each eligible document equal weight.",
    }


@app.local_entrypoint()
def main(output: str = "artifacts/repetition-baselines/corpus-patterns.json"):
    result = audit.remote(time.time() + LAUNCH_LIMIT_SECONDS)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print("audit complete", flush=True)
