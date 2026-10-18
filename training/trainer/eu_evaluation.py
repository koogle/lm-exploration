"""Matched, bounded before/after continuation checks; all artifacts stay private."""

import json
import sqlite3
import zlib
from pathlib import Path


def evaluate(checkpoint, eu_manifest, original_manifest, output):
    import torch
    from model_100m.model import GPT, ModelConfig
    from training_data.raw_token_stream import RawTokenMixture
    from training_data.subword import BOS_TOKEN_ID, EOS_TOKEN_ID, load_frozen

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = load_frozen()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = (
        GPT(
            ModelConfig(
                vocab_size=32768,
                block_size=config["context"],
                n_layer=config["n_layer"],
                n_head=config["n_head"],
                n_embd=config["n_embd"],
            )
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(payload["model"], strict=True)
    del payload
    eu_manifest = Path(eu_manifest)
    manifest = json.loads(eu_manifest.read_text())
    db = sqlite3.connect(
        f"file:{eu_manifest.parent / manifest['database']['path']}?mode=ro", uri=True
    )
    batches = {}
    for institution in ("ecb", "ema"):
        rows = db.execute(
            "SELECT text FROM documents WHERE conflict=0 AND split='validation' "
            "AND json_extract(provenance,'$.institution')=? ORDER BY hash LIMIT 32",
            (institution,),
        ).fetchall()
        windows = []
        for (raw,) in rows:
            ids = [
                BOS_TOKEN_ID,
                *tokenizer.encode(zlib.decompress(raw).decode(), add_special_tokens=False).ids,
                EOS_TOKEN_ID,
            ]
            windows.append((ids[:2048], ids[1:2049]))
        batches[institution] = windows
    db.close()
    original = RawTokenMixture(
        original_manifest, context=2048, split="validation", allow_unreviewed=True
    )
    for name in original.sources:
        blocks = []
        for block in original.blocks(name):
            count = block["loss_tokens"]
            blocks.append((block["input_ids"][:count], block["labels"][:count]))
            if len(blocks) == 4:
                break
        batches["original/" + name] = blocks
    original.close()
    losses = {}
    with torch.inference_mode():
        for name, windows in batches.items():
            total = 0.0
            tokens = 0
            for x, y in windows:
                x = x[: len(y)]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(torch.tensor([x], device="cuda"))
                    loss = torch.nn.functional.cross_entropy(
                        logits.flatten(0, 1), torch.tensor(y, device="cuda"), reduction="sum"
                    )
                total += float(loss)
                tokens += len(y)
            losses[name] = {
                "loss": total / tokens if tokens else None,
                "tokens": tokens,
                "windows": len(windows),
            }
        prompts = [
            "The solar system consists of",
            "A prime number is",
            "Once upon a time",
            "The capital of France is",
            "2 + 2 =",
            "def add(a, b):\n    ",
            "def factorial(n):\n    ",
            "The European Central Bank is responsible for",
            "Inflation means",
            "The European Medicines Agency evaluates",
        ]
        completions = []
        for prompt in prompts:
            prefix = [BOS_TOKEN_ID, *tokenizer.encode(prompt, add_special_tokens=False).ids]
            ids = torch.tensor([prefix], device="cuda")
            generated = []
            # Full FP32 generation avoids low-precision ties confounding comparisons.
            for _ in range(128):
                token = int(model(ids)[0, -1].argmax())
                generated.append(token)
                ids = torch.cat((ids, torch.tensor([[token]], device="cuda")), dim=1)
                if token == EOS_TOKEN_ID:
                    break
            grams = [tuple(generated[i : i + 4]) for i in range(max(0, len(generated) - 3))]
            completions.append(
                {
                    "prompt": prompt,
                    "text": tokenizer.decode(generated),
                    "generated_ids": generated,
                    "repeated_token_4gram_fraction": 1 - len(set(grams)) / len(grams)
                    if grams
                    else 0,
                }
            )
    result = {
        "checkpoint": str(checkpoint),
        "losses": losses,
        "completions": completions,
        "note": "Bounded fixed document-prefix diagnostic, not whole-corpus perplexity. "
        "EU: up to 32 validation parents per institution; original: 4 windows/source. "
        "Teacher-forced loss BF16; generation FP32, TF32 disabled, greedy, 128 tokens.",
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return {"checkpoint": str(checkpoint), "losses": losses, "artifact": str(output)}
