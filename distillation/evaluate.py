"""Compare existing checkpoints on the same larger, document-capped holdout."""

import hashlib
import json


def select_windows(data, max_documents=512, windows_per_document=8):
    if max_documents < 1 or windows_per_document < 1:
        raise ValueError("Evaluation limits must be positive")
    counts = {}
    windows = []
    while True:
        try:
            x, y = data.next()
        except StopIteration:
            break
        key = data.state["hash"]
        if key not in counts:
            if len(counts) == max_documents:
                break
            counts[key] = 0
        if counts[key] < windows_per_document:
            windows.append((key, x, y))
            counts[key] += 1
    if not windows:
        raise ValueError("Empty evaluation split")
    return windows


def compare(checkpoints):
    import torch
    import torch.nn.functional as F
    from model_100m.model import GPT, ModelConfig
    from training_data.subword import TOKENIZER_SHA256, load_frozen

    from distillation.data import CodeWindows
    from distillation.train import DATABASE, MANIFEST

    data = CodeWindows(
        MANIFEST, DATABASE, load_frozen(), context=256, split="validation"
    )
    windows = select_windows(data)
    identity = hashlib.sha256(
        json.dumps(windows, separators=(",", ":")).encode()
    ).hexdigest()
    results = []
    for checkpoint in checkpoints:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("format") == "byte-distillation-v1":
            config = dict(saved["model_config"])
            tokenizer_hash = saved["settings"]["tokenizer_sha256"]
        elif saved.get("format") == "scholarly-ddp-v1":
            config = {k: saved["config"][k] for k in ("n_layer", "n_head", "n_embd")}
            config.update(vocab_size=32768, block_size=256)
            tokenizer_hash = saved["data_identity"]["tokenizer_sha256"]
        else:
            raise ValueError("Unsupported checkpoint format")
        if tokenizer_hash != TOKENIZER_SHA256:
            raise ValueError("Cannot compare token CE across different tokenizers")
        config["block_size"] = 256
        model = GPT(ModelConfig(**config)).cuda().eval()
        model.load_state_dict(saved["model"], strict=True)
        del saved
        total_loss, tokens = 0.0, 0
        documents = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for key, x, y in windows:
                logits = model(torch.tensor([x], device="cuda"))[0]
                loss = F.cross_entropy(
                    logits.float(), torch.tensor(y, device="cuda"), reduction="sum"
                ).item()
                total_loss += loss
                tokens += len(y)
                doc = documents.setdefault(key, {"loss_sum": 0.0, "tokens": 0})
                doc["loss_sum"] += loss
                doc["tokens"] += len(y)
        results.append(
            {
                "checkpoint": checkpoint,
                "ce": total_loss / tokens,
                "tokens": tokens,
                "documents": documents,
            }
        )
        del model
        torch.cuda.empty_cache()
    return {
        "context": 256,
        "dtype": "bfloat16",
        "manifest_sha256": data.identity,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "window_sha256": identity,
        "windows": len(windows),
        "max_documents": 512,
        "windows_per_document": 8,
        "results": results,
        "torch_version": str(torch.__version__),
    }
