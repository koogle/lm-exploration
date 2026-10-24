"""Fit a BPE tokenizer on a weighted sample of a corpus recipe's training split.

Sample once, then fit and evaluate offline. Fitting text comes only from training
rows; evaluation text only from validation rows. Run from `training/`:

    uv run --extra corpus python -m tokenizer.mixture_tokenizer sample \
        --recipe data/recipes/climbmix-code-v1.json --output runs/tokenizer-sample
    uv run --extra corpus python -m tokenizer.mixture_tokenizer fit \
        --sample runs/tokenizer-sample --output tokenizer/climbmix-code-v1
    uv run --extra corpus python -m tokenizer.mixture_tokenizer evaluate \
        --sample runs/tokenizer-sample --tokenizer tokenizer/climbmix-code-v1 \
        --tokenizer tokenizer/structured-code-v1
"""

import argparse
import json
import time
from pathlib import Path

from tokenizers import Tokenizer, trainers
from tokenizers.pre_tokenizers import ByteLevel
from training_data.simple_llm import adapt, load_recipe, stream_rows
from training_data.subword import SPECIAL_TOKEN_IDS, digest

from tokenizer.tokenizer_research import new_tokenizer

# GPT-2-style splitting, except every digit is its own pre-token so arithmetic
# sees one token per digit.
PRETOKENIZATION_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def sample(recipe_path, output, train_bytes, eval_bytes, files_per_source=64, reader=stream_rows):
    """Spread each source's byte quota evenly over its first `files_per_source` shards."""
    recipe = load_recipe(recipe_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    with (output / "train.jsonl").open("w") as train, (output / "eval.jsonl").open("w") as held:
        for source in recipe["sources"]:
            files = source["files"][:files_per_source]
            share = source["weight_percent"] / 100
            caps = {
                "train": share * train_bytes / len(files),
                "validation": share * eval_bytes / len(files),
            }
            totals = {"train": 0, "validation": 0, "documents": 0}
            for shard in files:
                taken = {"train": 0, "validation": 0}
                rows = reader(source, shard, 0)
                try:
                    for index, row in rows:
                        document, _ = adapt(source, row, (shard, index))
                        if document is None or document["split"] not in taken:
                            continue
                        split = document["split"]
                        if taken[split] >= caps[split]:
                            if all(taken[s] >= caps[s] for s in taken):
                                break
                            continue
                        size = len(document["text"].encode())
                        handle = train if split == "train" else held
                        handle.write(
                            json.dumps({"source": source["name"], "text": document["text"]})
                        )
                        handle.write("\n")
                        taken[split] += size
                        totals[split] += size
                        totals["documents"] += 1
                finally:
                    rows.close()
            counts[source["name"]] = {**totals, "files": len(files)}
            print(json.dumps({"source": source["name"], **counts[source["name"]]}), flush=True)
    manifest = {
        "recipe": recipe,
        "recipe_sha256": digest(recipe_path),
        "train_bytes_target": train_bytes,
        "eval_bytes_target": eval_bytes,
        "files_per_source": files_per_source,
        "sources": counts,
        "train_sha256": digest(output / "train.jsonl"),
        "eval_sha256": digest(output / "eval.jsonl"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def fit(sample_folder, output, vocab=32768):
    sample_folder, output = Path(sample_folder), Path(output)
    manifest = json.loads((sample_folder / "manifest.json").read_text())
    if digest(sample_folder / "train.jsonl") != manifest["train_sha256"]:
        raise ValueError("Fitting sample hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "format": "huggingface-tokenizers-json",
        "vocab_size": vocab,
        "special_tokens": SPECIAL_TOKEN_IDS,
        "normalization": None,
        "pretokenization_regex": PRETOKENIZATION_REGEX,
        "add_prefix_space": False,
        "automatic_bos_eos": False,
        "padding": None,
        "truncation": None,
    }
    tokenizer = new_tokenizer(config)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        min_frequency=2,
        max_token_length=32,
        special_tokens=list(SPECIAL_TOKEN_IDS),
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    start = time.perf_counter()
    tokenizer.train_from_iterator((r["text"] for r in rows(sample_folder / "train.jsonl")), trainer)
    elapsed = time.perf_counter() - start
    tokenizer.save(str(output / "tokenizer.json"), pretty=False)
    if tokenizer.get_vocab_size() != vocab:
        raise ValueError("Requested vocabulary not reached")
    for spelling, expected in SPECIAL_TOKEN_IDS.items():
        if tokenizer.token_to_id(spelling) != expected:
            raise ValueError("Unexpected special token IDs")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sample_record = {k: v for k, v in manifest.items() if k != "recipe"}
    sample_record["recipe_id"] = manifest["recipe"]["id"]
    sample_record["fit_seconds"] = elapsed
    (output / "sample.json").write_text(json.dumps(sample_record, indent=2) + "\n")
    return sample_record


def evaluate(sample_folder, tokenizer_folders):
    """Bytes per token and exact round trips on the sample's validation rows."""
    texts = list(rows(Path(sample_folder) / "eval.jsonl"))
    results = {}
    for folder in tokenizer_folders:
        tokenizer = Tokenizer.from_file(str(Path(folder) / "tokenizer.json"))
        metrics = {}
        for row in texts:
            ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
            m = metrics.setdefault(
                row["source"], {"bytes": 0, "tokens": 0, "roundtrip_failures": 0}
            )
            m["bytes"] += len(row["text"].encode())
            m["tokens"] += len(ids)
            m["roundtrip_failures"] += (
                tokenizer.decode(ids, skip_special_tokens=False) != row["text"]
            )
        for m in metrics.values():
            m["bytes_per_token"] = m["bytes"] / m["tokens"]
        results[Path(folder).name] = {
            "tokenizer_sha256": digest(Path(folder) / "tokenizer.json"),
            "sources": metrics,
            "digits": tokenizer.encode("12345 + 678 = 13023").tokens,
        }
    return {"eval_sha256": digest(Path(sample_folder) / "eval.jsonl"), "tokenizers": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    s = commands.add_parser("sample")
    s.add_argument("--recipe", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--train-mib", type=int, default=256)
    s.add_argument("--eval-mib", type=int, default=16)
    f = commands.add_parser("fit")
    f.add_argument("--sample", required=True)
    f.add_argument("--output", required=True)
    e = commands.add_parser("evaluate")
    e.add_argument("--sample", required=True)
    e.add_argument("--tokenizer", action="append", required=True)
    e.add_argument("--output")
    args = parser.parse_args()
    if args.command == "sample":
        sample(args.recipe, args.output, args.train_mib * 1024**2, args.eval_mib * 1024**2)
    elif args.command == "fit":
        print(json.dumps(fit(args.sample, args.output), indent=2))
    else:
        report = evaluate(args.sample, args.tokenizer)
        text = json.dumps(report, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(text)
        print(text)


if __name__ == "__main__":
    main()
