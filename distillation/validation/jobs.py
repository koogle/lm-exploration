"""Container job bodies: student training/evaluation and campaign data helpers."""

import json
import time
from pathlib import Path

import numpy as np
from common import QWEN_ENDOFTEXT, QWEN_VOCAB, atomic_json

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


def vllm_config(saved_config, bos, eos):
    """config.json for vLLM: Llama-typed so it parses, naming our decoder class."""
    width, heads = saved_config["n_embd"], saved_config["n_head"]
    return {
        "architectures": ["SimpleLLMForCausalLM"],  # vllm_model.ARCHITECTURE
        "model_type": "llama",
        "vocab_size": saved_config["vocab_size"],
        "hidden_size": width,
        "intermediate_size": (8 * width + 2) // 3,
        "num_hidden_layers": saved_config["n_layer"],
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "head_dim": width // heads,
        "max_position_embeddings": saved_config["block_size"],
        "tie_word_embeddings": True,
        "bos_token_id": bos,
        "eos_token_id": eos,
        "torch_dtype": "float32",
    }


def vllm_model_dir(checkpoint, vocab, out_dir):
    """A vLLM model directory around the training checkpoint (linked, not converted)."""
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    eos = int(vocab.encode([IM_END])[0])
    atomic_json(out_dir / "config.json", vllm_config(saved["config"], vocab.bos, eos))
    link = out_dir / "model.pt"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(checkpoint).resolve())
    arrays = {
        "table": vocab.table,
        "native": vocab.native,
        "bos": np.asarray(vocab.bos),
    }
    if vocab.pieces is not None:
        arrays.update(pieces=vocab.pieces, offsets=vocab.offsets, lengths=vocab.lengths)
    np.savez(out_dir / "vocab.npz", **arrays)
    return out_dir


def student_run(settings, manifest, items, run_dir, deadline, eval_settings):
    """Train (or load ``settings["checkpoint"]``), then evaluate the checkpoint with vLLM."""
    import subprocess

    import torch
    from student import train, validate

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "settings.json", settings)
    vocab = load_vocab(manifest, settings["vocab"])
    if settings.get("checkpoint"):
        checkpoint = Path("/runs") / settings["checkpoint"]
        summary = {
            **json.loads((checkpoint.parent / "summary.json").read_text()),
            "source_checkpoint": settings["checkpoint"],
        }
    else:
        log_file = (run_dir / "metrics.jsonl").open("a")

        def log(entry):
            entry = {"time": time.time(), **entry}
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            print(json.dumps(entry), flush=True)

        torch.backends.cuda.matmul.allow_tf32 = True
        eos = int(vocab.encode([IM_END])[0])
        model, summary = train(
            settings, manifest["train"], vocab, eos, run_dir, log, deadline
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
        )
        summary["validation_seconds"] = time.time() - started
        log({"event": "validation", **summary["validation"]})
        log_file.close()
        checkpoint = run_dir / "model.pt"
        del model
        torch.cuda.empty_cache()
    model_dir = vllm_model_dir(checkpoint, vocab, run_dir / "vllm")
    items_path, out = run_dir / "eval-items.json", run_dir / "vllm.json"
    items_path.write_text(json.dumps(items))
    subprocess.run(
        [
            "python",
            "/validation/vllm_eval.py",
            "student",
            str(items_path),
            str(out),
            "--model-dir",
            str(model_dir),
            "--tokenizer",
            manifest["tokenizer"],
            "--max-tokens",
            str(eval_settings["max_new_tokens"]),
        ],
        cwd="/validation",
        check=True,
    )
    report = json.loads(out.read_text())
    summary.update(parity=report["parity"], generation_seconds=report["seconds"])
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "completions.json", report["results"])
    return {"summary": summary, "completions": report["results"]}


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
