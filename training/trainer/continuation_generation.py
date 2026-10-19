"""Matched checkpoint generation; raw samples stay on the private run volume."""

import json
import time
from pathlib import Path

PROMPTS = [
    ("general", "The solar system consists of"),
    ("math", "A prime number is"),
    ("story", "Once upon a time, in a small village,"),
    ("math", "The derivative of x squared is"),
    ("code", "def add(a, b):\n    "),
    ("code", "def factorial(n):\n    "),
    ("code", "def is_palindrome(text):\n    "),
    ("ecb", "The European Central Bank is responsible for"),
    ("ecb", "Inflation means"),
    ("ema", "The European Medicines Agency evaluates"),
]
MODES = {
    "greedy": {"temperature": 0.0, "top_k": 0, "seeds": [42]},
    "low_temperature_topk": {"temperature": 0.2, "top_k": 10, "seeds": [42, 43]},
    "sampled_topk": {"temperature": 0.8, "top_k": 50, "seeds": [42, 43]},
}


def repetition_metrics(ids):
    """Generated-only repeats; loop flag means >=64 tokens in >=3 tandem copies."""
    grams = [tuple(ids[i : i + 4]) for i in range(max(0, len(ids) - 3))]
    longest = 0
    for period in range(1, min(64, len(ids) // 3) + 1):
        matched = 0
        for i in range(period, len(ids)):
            matched = matched + 1 if ids[i] == ids[i - period] else 0
            if matched >= 2 * period:
                longest = max(longest, matched + period)
    return {
        "repeated_token_4gram_fraction": 1 - len(set(grams)) / len(grams) if grams else 0.0,
        "longest_tandem_repeat_tokens": longest,
        "long_loop": longest >= 64,
    }


def generate_checkpoint(checkpoint, output, label, commit=lambda: None, max_new_tokens=256):
    import torch
    from model_100m.model import GPT, ModelConfig
    from training_data.subword import (
        BOS_TOKEN_ID,
        EOS_TOKEN_ID,
        PAD_TOKEN_ID,
        UNK_TOKEN_ID,
        digest,
        load_frozen,
    )

    from trainer.distributed_train import atomic_json

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise ValueError("Refuse to overwrite generation samples")
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
    identity = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "phase_tokens": payload["tokens"],
        "step": payload["step"],
        "lifetime_tokens": payload.get("lineage", {}).get("parent_tokens", 0) + payload["tokens"],
    }
    del payload
    tokenizer = load_frozen()
    result = {
        "label": label,
        "identity": identity,
        "max_new_tokens": max_new_tokens,
        "precision": "FP32; TF32 disabled; no repetition penalty/ngram blocking",
        "special_tokens": "PAD/BOS/UNK suppressed; stop on EOS",
        "prompts": PROMPTS,
        "modes": MODES,
        "samples": [],
        "completed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for mode, settings in MODES.items():
            for seed in settings["seeds"]:
                for index, (category, prompt) in enumerate(PROMPTS):
                    # The same independent RNG is reset for each matched prompt.
                    generator = torch.Generator(device="cuda").manual_seed(seed + index * 1000)
                    prefix = [BOS_TOKEN_ID, *tokenizer.encode(prompt, add_special_tokens=False).ids]
                    context = torch.tensor([prefix], device="cuda")
                    generated = []
                    stopped = "token_limit"
                    for _ in range(max_new_tokens):
                        logits = model(context[:, -config["context"] :])[0, -1].float()
                        if not torch.isfinite(logits).all():
                            raise ValueError("Nonfinite logits")
                        logits[[PAD_TOKEN_ID, BOS_TOKEN_ID, UNK_TOKEN_ID]] = -torch.inf
                        if settings["temperature"] == 0:
                            token = int(logits.argmax())
                        else:
                            values, indices = torch.topk(
                                logits / settings["temperature"], settings["top_k"]
                            )
                            token = int(
                                indices[
                                    torch.multinomial(values.softmax(-1), 1, generator=generator)
                                ]
                            )
                        if token == EOS_TOKEN_ID:
                            stopped = "eos"
                            break
                        generated.append(token)
                        context = torch.cat(
                            (context, torch.tensor([[token]], device="cuda")), dim=1
                        )
                    result["samples"].append(
                        {
                            "mode": mode,
                            "seed": seed,
                            "prompt_index": index,
                            "category": category,
                            "prompt": prompt,
                            "text": tokenizer.decode(generated),
                            "generated_ids": generated,
                            "new_tokens": len(generated),
                            "stop_reason": stopped,
                            **repetition_metrics(generated),
                        }
                    )
                atomic_json(output, result)
                commit()
                print(json.dumps({"label": label, "samples": len(result["samples"])}), flush=True)
    result.update(
        completed=True,
        elapsed_seconds=time.monotonic() - started,
        torch_version=str(torch.__version__),
        gpu=torch.cuda.get_device_name(),
    )
    atomic_json(output, result)
    commit()
    return {
        "completed": True,
        "label": label,
        "samples": len(result["samples"]),
        "elapsed_seconds": result["elapsed_seconds"],
        "artifact": str(output),
    }
