"""Container job bodies: student training/evaluation and campaign data helpers."""

import json
import time
from pathlib import Path

import numpy as np
from common import QWEN_ENDOFTEXT, QWEN_VOCAB, atomic_json
from evalsets import parse_code_string, student_prompt

IM_END = 248046  # teacher stop token at the end of every accepted document


def load_vocab(manifest, kind):
    from student import Vocabulary

    if kind == "full":
        return Vocabulary.full(QWEN_VOCAB, QWEN_ENDOFTEXT)
    if kind == "covering":
        from prepare import build_covering_vocab

        counts = sum(np.load(Path(shard) / "counts.npy") for shard in manifest["train"])
        tokenizer = json.loads(Path(manifest["tokenizer"]).read_text())
        size = manifest["trimmed_vocab"]
        return Vocabulary.covering(*build_covering_vocab(counts, size, tokenizer))
    return Vocabulary.trimmed(
        np.load(manifest["vocab_table"]), np.load(manifest["vocab_kept"])
    )


def complete(model, vocab, tokenizer, items, max_new, batch, device="cuda"):
    """Greedy training-format completions; returns {id: {text, code, unknown}}."""
    from student import UNK, generate

    eos = int(vocab.encode([IM_END])[0])
    prompts = {
        item["id"]: [vocab.bos]
        + vocab.expand(
            tokenizer.encode(
                student_prompt(item["problem"]), add_special_tokens=False
            ).ids
        )[0].tolist()
        for item in items
    }
    order = sorted(prompts, key=lambda key: len(prompts[key]))

    def text_of(ids):
        return tokenizer.decode(vocab.decode_native(ids), skip_special_tokens=False)

    def finished(ids):
        return parse_code_string(text_of(ids)) is not None

    trimmed = vocab.size < QWEN_VOCAB
    results = {}
    for start in range(0, len(order), batch):
        keys = order[start : start + batch]
        outputs = generate(
            model, [prompts[k] for k in keys], max_new, eos, finished, device=device
        )
        for key, ids in zip(keys, outputs):
            text = text_of(ids)
            results[key] = {
                "text": text,
                "code": parse_code_string(text),
                "unknown_tokens": sum(1 for i in ids if i == UNK) if trimmed else 0,
                "prompt_unknown_tokens": prompts[key].count(UNK) if trimmed else 0,
                "generated_tokens": len(ids),
            }
    return results


def student_run(
    settings, manifest, items, run_dir, deadline, eval_settings, device="cuda"
):
    import torch
    from student import train, validate
    from tokenizers import Tokenizer

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "settings.json", settings)
    vocab = load_vocab(manifest, settings["vocab"])
    log_file = (run_dir / "metrics.jsonl").open("a")

    def log(entry):
        entry = {"time": time.time(), **entry}
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()
        print(json.dumps(entry), flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    model, summary = train(
        settings, manifest["train"], vocab, run_dir, log, deadline, device=device
    )
    summary["vocabulary_size"] = vocab.size
    started = time.time()
    summary["validation"] = validate(
        model,
        manifest["heldout"],
        vocab,
        np.load(manifest["byte_lengths"]),
        settings["validation_positions"],
        window=settings["window"],
        device=device,
    )
    summary["validation_seconds"] = time.time() - started
    log({"event": "validation", **summary["validation"]})
    tokenizer = Tokenizer.from_file(manifest["tokenizer"])
    started = time.time()
    completions = complete(
        model,
        vocab,
        tokenizer,
        items,
        eval_settings["max_new_tokens"],
        eval_settings["generation_batch"],
        device=device,
    )
    summary["generation_seconds"] = time.time() - started
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "completions.json", completions)
    log_file.close()
    return {"summary": summary, "completions": completions}


def wilson(passed, total, z=1.96):
    """95% Wilson score interval for a pass rate."""
    if not total:
        return [0.0, 0.0]
    p = passed / total
    centre = p + z * z / (2 * total)
    spread = z * ((p * (1 - p) + z * z / (4 * total)) / total) ** 0.5
    scale = 1 + z * z / total
    return [(centre - spread) / scale, (centre + spread) / scale]


def score(items, completions, grades):
    """Pass counts per eval set for one model's completions."""
    sets = {}
    for item in items:
        entry = sets.setdefault(item["set"], {"total": 0, "passed": 0, "parsed": 0})
        entry["total"] += 1
        completion = completions.get(item["id"]) or {}
        if completion.get("code"):
            entry["parsed"] += 1
        if grades.get(item["id"], {}).get("passed"):
            entry["passed"] += 1
    for entry in sets.values():
        entry["pass_rate"] = entry["passed"] / entry["total"]
        entry["pass_rate_95ci"] = wilson(entry["passed"], entry["total"])
    return sets
