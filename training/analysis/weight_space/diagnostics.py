"""Diagnostic measurements on frozen weights and stored training examples."""

import json
import time
from collections import Counter
from pathlib import Path


def quantiles(values):
    import numpy as np

    if not len(values):
        return {}
    return {str(q): float(np.quantile(values, q)) for q in (0, 0.5, 0.9, 0.95, 0.99, 1)}


def audit_training(out, deadline):
    import numpy as np
    from training_data.qwen_agent_chat import parse_assistant
    from transformers import AutoTokenizer

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path("/runs/agent-mix/scale-500k")
    manifest = json.loads((root / "manifest.json").read_text())
    offsets, order = np.load(root / "offsets.npy"), np.load(root / "order.npy")
    tokens = np.memmap(root / "tokens.u32", dtype=np.uint32, mode="r")
    masks = np.memmap(root / "mask.u8", dtype=np.uint8, mode="r")
    assert offsets[-1] == len(tokens) == len(masks)
    tokenizer = AutoTokenizer.from_pretrained(
        "/runs/agentic-reference/qwen3-0.6b-base-scale-500k-02/model"
    )
    tokenizer_check = {}
    try:
        train_tokenizer = AutoTokenizer.from_pretrained(
            manifest["tokenizer"][0], revision=manifest["tokenizer"][1], local_files_only=True
        )
        tokenizer_check["vocabulary_equal"] = tokenizer.get_vocab() == train_tokenizer.get_vocab()
    except OSError as error:
        tokenizer_check["unavailable"] = type(error).__name__
    result = {
        "manifest": manifest,
        "tokenizer_check": tokenizer_check,
        "lengths": quantiles(np.diff(offsets)[order]),
        "sources": {},
    }
    rng = np.random.default_rng(20261103)
    examples = []
    for source, part in manifest["counts"].items():
        first, end = part["stored"]
        selected = rng.choice(np.arange(first, end), min(1000, end - first), replace=False)
        counts, lengths, repeat_examples = Counter(), [], []
        for index in selected:
            if time.time() > deadline - 120:
                raise TimeoutError("Training audit deadline")
            start, stop = offsets[index : index + 2]
            ids, mask = tokens[start:stop], masks[start:stop]
            counts["examples"] += 1
            counts["nonbinary_masks"] += int(((mask != 0) & (mask != 1)).sum())
            edges = np.diff(np.r_[0, mask, 0].astype(np.int16))
            begins, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            seen = set()
            commands = []
            for lo, hi in zip(begins, ends, strict=True):
                length = int(hi - lo)
                lengths.append(length)
                counts["assistant_spans"] += 1
                counts["span_over_1536"] += length > 1536
                counts["span_over_4096"] += length > 4096
                counts["span_without_im_end"] += int(ids[hi - 1]) != 151645
                counts["im_start_in_supervision"] += int((ids[lo:hi] == 151644).sum())
                text = tokenizer.decode(ids[lo:hi])
                calls = parse_assistant(text).get("tool_calls", [])
                if "<tool_call>" in text and not calls:
                    counts["unparseable_tool_spans"] += 1
                for call in calls:
                    if call["function"]["name"] != "bash":
                        counts["non_bash_calls"] += 1
                        continue
                    command = json.loads(call["function"]["arguments"]).get("command")
                    if not isinstance(command, str):
                        counts["invalid_command_types"] += 1
                        continue
                    counts["commands"] += 1
                    counts["exact_repeated_commands"] += command in seen
                    counts["commands_10plus_grep_exclusions"] += command.count("grep -v") >= 10
                    seen.add(command)
                    commands.append(command)
                if len(examples) < 30 and source in ("mini-coder", "open-swe") and length > 1536:
                    examples.append(
                        {"source": source, "index": int(index), "tokens": length, "assistant": text}
                    )
            if len(commands) != len(set(commands)):
                repeat_examples.append(int(index))
        result["sources"][source] = {
            "counts": dict(counts),
            "assistant_token_lengths": quantiles(lengths),
            "repeating_examples": len(repeat_examples),
            "example_lengths": quantiles(np.diff(offsets)[first:end]),
            "sample_indices": selected.tolist(),
        }
        (out / "training-audit.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({source: result["sources"][source]["counts"]}), flush=True)
    (out / "long-training-outputs.json").write_text(json.dumps(examples, indent=2))
    # Count verified candidate metadata only from locally cached source shards.
    from huggingface_hub import hf_hub_download
    from pyarrow import parquet

    from modal_scripts.modal_agent_sft import SOURCES, excluded, mini_coder_repo

    spec = SOURCES["mini-coder"]
    quality = Counter()
    missing = []
    for shard in range(60):
        name = f"data/train-{shard:05d}-of-00060.parquet"
        try:
            path = hf_hub_download(
                spec["repo"],
                name,
                repo_type="dataset",
                revision=spec["revision"],
                local_files_only=True,
            )
        except OSError:
            missing.append(name)
            continue
        for row in parquet.read_table(path, columns=["instance_id", "verified"]).to_pylist():
            if not excluded(mini_coder_repo(row["instance_id"])):
                quality["eligible"] += 1
                quality["verified"] += bool(row["verified"])
    result["mini_coder_candidate_quality"] = {
        **quality,
        "missing_shards": missing,
        "scope": "eligible raw candidates before conversion/context filtering; stored membership is not mapped back to raw rows",
    }
    (out / "training-audit.json").write_text(json.dumps(result, indent=2))
    return result
