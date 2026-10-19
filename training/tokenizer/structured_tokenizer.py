"""Controlled code/JSON/diff/tool-call tokenizer experiment (not model evaluation)."""

import difflib
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer, trainers
from tokenizers.pre_tokenizers import ByteLevel
from training_data.subword import digest

from tokenizer.tokenizer_research import new_tokenizer

CATEGORIES = ("python", "json", "git_diff", "tool_call")
WEIGHTS = dict(zip(CATEGORIES, (40, 20, 20, 20), strict=True))


def examples(row):
    """Deterministic fixtures derived from a real source excerpt; no execution."""
    text = row["text"]
    identity = hashlib.sha256(row["parent"].encode()).hexdigest()
    number = int(identity[:8], 16)
    names = re.findall(r"\b(?:def|class)\s+([A-Za-z_]\w*)", text)[:12]
    path = f"src/{names[0] if names else 'module_' + identity[:8]}.py"
    # Four real edits with valid line-oriented unified patches.
    before = text if text.endswith("\n") else text + "\n"
    lines = before.splitlines(keepends=True)
    mode = number % 4
    if mode == 0:
        after = "# Updated implementation\n" + before
    elif mode == 1 and names:
        after = re.sub(r"\b" + re.escape(names[0]) + r"\b", names[0] + "_updated", before)
    elif mode == 2 and any(line.lstrip().startswith("#") for line in lines):
        removed = next(i for i, line in enumerate(lines) if line.lstrip().startswith("#"))
        after = "".join(lines[:removed] + lines[removed + 1 :])
    else:
        after = before + "\n# End of module\n"
    diff = f"diff --git a/{path} b/{path}\n" + "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    record = {
        "path": path,
        "symbols": names,
        "language": "python",
        "content": text,
        "metadata": {
            "enabled": True,
            "error": None,
            "version": number % 100,
            "score": (number % 10000) / 1000,
            "labels": ["code", "review"],
        },
    }
    # Compact vs pretty JSON, escapes, nested payloads, schemas, calls and results.
    kind = number % 4
    if kind == 0:
        tool = {
            "id": f"call_{identity[:16]}",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": path, "start_line": 1, "end_line": 120}),
            },
        }
    elif kind == 1:
        tool = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{identity[:16]}",
                    "type": "function",
                    "function": {"name": "apply_patch", "arguments": json.dumps({"patch": diff})},
                }
            ],
        }
    elif kind == 2:
        tool = {
            "role": "tool",
            "tool_call_id": f"call_{identity[:16]}",
            "content": json.dumps(record, ensure_ascii=False),
        }
    else:
        tool = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_code",
                        "description": f"Find definitions in {path}",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "paths": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_calls": [
                {
                    "name": "search_code",
                    "arguments": {"query": names[0] if names else "TODO", "paths": [path]},
                }
            ],
        }
    return {
        "python": text,
        "json": json.dumps(
            record, ensure_ascii=bool(number % 2), indent=2 if number % 3 == 0 else None
        ),
        "git_diff": diff,
        "tool_call": json.dumps(tool, ensure_ascii=False, separators=(",", ":")),
    }


def read_rows(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def build(training, validation, output, target_bytes=64 * 1024**2, eval_parents=512):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Unique source parents, fixed random subset independent of candidate scores.
    parents = {}
    for row in read_rows(validation):
        parents.setdefault(row["parent"], row)
    rows = list(parents.values())
    random.Random(20261016).shuffle(rows)
    rows = rows[:eval_parents]
    if len(rows) != eval_parents:
        raise ValueError("Insufficient held-out parents")
    held_out = {r["parent"] for r in rows}
    evaluation_hashes = set()
    with (output / "evaluation.jsonl").open("w") as handle:
        for row in rows:
            for category, text in examples(row).items():
                evaluation_hashes.add(hashlib.sha256(text.encode()).hexdigest())
                handle.write(
                    json.dumps({"category": category, "parent": row["parent"], "text": text}) + "\n"
                )
    totals, counts, fitting_parents = Counter(), Counter(), set()
    seen = set()
    with (output / "train.jsonl").open("w") as handle:
        for row in read_rows(training):
            if row["parent"] in held_out:
                raise ValueError("Fitting/evaluation parent overlap")
            for category, text in examples(row).items():
                if totals[category] >= target_bytes * WEIGHTS[category] // 100:
                    continue
                key = hashlib.sha256(text.encode()).hexdigest()
                if key in seen or key in evaluation_hashes:
                    continue
                seen.add(key)
                fitting_parents.add(row["parent"])
                totals[category] += len(text.encode())
                counts[category] += 1
                handle.write(
                    json.dumps({"category": category, "parent": row["parent"], "text": text}) + "\n"
                )
            if all(totals[c] >= target_bytes * WEIGHTS[c] // 100 for c in CATEGORIES):
                break
        else:
            raise ValueError("Insufficient fitting text without repetition")
    manifest = {
        "seed": 20261016,
        "target_bytes": target_bytes,
        "weights_percent": WEIGHTS,
        "actual_bytes": dict(totals),
        "training_examples": dict(counts),
        "fitting_parents": len(fitting_parents),
        "evaluation_parents_per_category": eval_parents,
        "train_sha256": digest(output / "train.jsonl"),
        "evaluation_sha256": digest(output / "evaluation.jsonl"),
        "source_train_sha256": digest(training),
        "source_validation_sha256": digest(validation),
        "implementation_sha256": digest(__file__),
        "limitations": [
            "Python excerpts from the existing pinned codeparrot corpus, not necessarily complete parseable files",
            "JSON, tool calls and patches are generated controlled fixtures, not independent production traces",
            "Fitting/evaluation parents are disjoint; generator templates are shared",
            "Each category quota includes its final whole example, so 64 MiB is approximate",
            "Unified diffs cover insertions, deletions and identifier replacements, but not binary patches or renames",
            "One fitting seed; no model-level tool-call or code-generation correctness measured",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def fit(folder, output, vocab, config_path="/project/tokenizer/simple-llm-v1/config.json"):
    folder, output = Path(folder), Path(output)
    manifest = json.loads((folder / "manifest.json").read_text())
    if digest(folder / "train.jsonl") != manifest["train_sha256"]:
        raise ValueError("Fitting data hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(config_path).read_text())
    tokenizer = new_tokenizer(config)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        min_frequency=2,
        max_token_length=32,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    start = time.perf_counter()
    tokenizer.train_from_iterator((r["text"] for r in read_rows(folder / "train.jsonl")), trainer)
    elapsed = time.perf_counter() - start
    tokenizer.save(str(output / "tokenizer.json"))
    if tokenizer.get_vocab_size() != vocab:
        raise ValueError("Requested vocabulary not reached")
    metadata = {
        "vocab": vocab,
        "training_seconds": elapsed,
        "tokenizer_sha256": digest(output / "tokenizer.json"),
        "corpus_manifest_sha256": digest(folder / "manifest.json"),
        "execution": "modal-cpu",
    }
    (output / "fit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def score(tokenizer_path, evaluation, name):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    metrics = {}
    for row in read_rows(evaluation):
        text = row["text"]
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        m = metrics.setdefault(
            row["category"],
            {
                "bytes": 0,
                "tokens": 0,
                "examples": 0,
                "roundtrip_failures": 0,
                "unk_tokens": 0,
                "per_parent": [],
            },
        )
        nbytes = len(text.encode())
        m["bytes"] += nbytes
        m["tokens"] += len(ids)
        m["examples"] += 1
        m["roundtrip_failures"] += tokenizer.decode(ids, skip_special_tokens=False) != text
        m["unk_tokens"] += ids.count(3)
        m["per_parent"].append(
            {
                "parent_sha256": hashlib.sha256(row["parent"].encode()).hexdigest(),
                "bytes": nbytes,
                "tokens": len(ids),
            }
        )
    for m in metrics.values():
        m["bytes_per_token"] = m["bytes"] / m["tokens"]
        m["mean_tokens_per_example"] = m["tokens"] / m["examples"]
        lengths = sorted(p["tokens"] for p in m["per_parent"])
        m["p95_tokens_per_example"] = lengths[int(0.95 * (len(lengths) - 1))]
    probes = [
        '{"name":"read_file","arguments":{"path":"src/main.py"}}',
        "diff --git a/main.py b/main.py\n@@ -1,2 +1,2 @@\n-old\n+new\n",
        "get_user_configuration",
        '"additionalProperties":false',
        '"tool_call_id":"call_0123456789"',
    ]
    return {
        "name": name,
        "vocab": tokenizer.get_vocab_size(),
        "execution": "modal-cpu",
        "tokenizer_sha256": digest(tokenizer_path),
        "evaluation_sha256": digest(evaluation),
        "metrics": metrics,
        "fragmentation": {p: len(tokenizer.encode(p).ids) for p in probes},
        "correctness_passed": all(
            m["roundtrip_failures"] == 0 and m["unk_tokens"] == 0 for m in metrics.values()
        ),
    }
