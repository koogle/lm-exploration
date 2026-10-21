"""Fixed Python generations and hidden-test inputs; never executes generated code."""

import ast
import json
from pathlib import Path

from .continuation_generation import repetition_metrics

PROMPTS = [
    "def add(a, b):\n    ",
    "def factorial(n):\n    ",
    "def is_palindrome(text):\n    ",
    "def binary_search(items, target):\n    ",
    "def flatten(items):\n    ",
    "def count_words(text):\n    ",
    "def fibonacci(n):\n    ",
    "def test_add():\n    ",
]


def generate(model, tokenizer, prompt, *, temperature, seed, max_tokens=256):
    import torch
    from training_data.subword import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID

    prefix = [BOS_TOKEN_ID, *tokenizer.encode(prompt, add_special_tokens=False).ids]
    context = torch.tensor([prefix], device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    generated = []
    stopped = "token_limit"
    with torch.inference_mode():
        for _ in range(max_tokens):
            logits = model(context[:, -model.config.block_size :])[0, -1].float()
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite generation logits")
            logits[[PAD_TOKEN_ID, BOS_TOKEN_ID, UNK_TOKEN_ID]] = -torch.inf
            if temperature == 0:
                token = int(logits.argmax())
            else:
                values, indices = torch.topk(logits / temperature, 50)
                token = int(indices[torch.multinomial(values.softmax(-1), 1, generator=generator)])
            if token == EOS_TOKEN_ID:
                stopped = "eos"
                break
            generated.append(token)
            context = torch.cat((context, torch.tensor([[token]], device="cuda")), dim=1)
    completion = tokenizer.decode(generated, skip_special_tokens=False)
    try:
        ast.parse(prompt + completion)
        syntax_valid = True
    except (SyntaxError, ValueError, RecursionError):
        syntax_valid = False
    return {
        "prompt": prompt,
        "completion": completion,
        "token_ids": generated,
        "temperature": temperature,
        "top_k": 0 if temperature == 0 else 50,
        "seed": seed,
        "new_tokens": len(generated),
        "stop_reason": stopped,
        "syntax_valid": syntax_valid,
        **repetition_metrics(generated),
    }


def evaluate_checkpoint(checkpoint, task_path):
    import torch
    from model_100m.model import GPT, ModelConfig
    from training_data.subword import load_frozen

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = (
        GPT(
            ModelConfig(
                vocab_size=32768,
                block_size=cfg["context"],
                n_layer=cfg["n_layer"],
                n_head=cfg["n_head"],
                n_embd=cfg["n_embd"],
            )
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(payload["model"], strict=True)
    tokenizer = load_frozen()
    result = {
        "tokens": payload["tokens"],
        "step": payload["step"],
        "checkpoint": str(checkpoint),
        "precision": "FP32; TF32 disabled",
        "max_new_tokens": 256,
        "samples": [],
        "functions": [],
    }
    del payload
    for index, prompt in enumerate(PROMPTS):
        for temperature, seed in [(0.0, 42), (0.8, 42), (0.8, 43)]:
            result["samples"].append(
                generate(
                    model, tokenizer, prompt, temperature=temperature, seed=seed + 1000 * index
                )
            )
    tasks = json.loads(Path(task_path).read_text())
    result["task_dataset"] = {
        k: v for k, v in tasks.items() if k not in ("tasks", "reference_hashes")
    }
    for task in tasks["tasks"]:
        sample = generate(model, tokenizer, task["prompt"], temperature=0.0, seed=42)
        sample.update(
            task_id=task["task_id"], tests=task["tests"], test_imports=task["test_imports"]
        )
        result["functions"].append(sample)
    return result
