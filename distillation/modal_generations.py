"""Matched greedy code continuations from saved shared-tokenizer checkpoints."""

import json
import re
from pathlib import Path

import modal

from distillation.modal_app import artifacts, image

app = modal.App("lm-distillation-generations", include_source=False)
RUN = "qwen4b-shared-50m-20261021-01"
STEPS = [0, 147, 715, 1439, 3588, 7162]
PROMPTS = {
    "add": 'def add(a, b):\n    """Return the sum of a and b."""\n',
    "unique": 'def unique(items):\n    """Return a list with duplicates removed, preserving order."""\n',
    "factorial": 'def factorial(n):\n    """Return n factorial for a non-negative integer n."""\n',
}


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=900,
    retries=0,
    volumes={"/distill": artifacts},
)
def worker(output_id):
    import hashlib

    import torch
    from model_100m.model import GPT, ModelConfig
    from transformers import AutoTokenizer

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", output_id):
        raise ValueError("Invalid output ID")
    output = Path("/distill") / output_id
    output.mkdir(exist_ok=False)
    root = Path("/distill") / RUN
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    digest = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    eos = tokenizer.eos_token_id
    report = {
        "run": RUN,
        "decoding": "greedy",
        "temperature": 0,
        "max_new_tokens": 128,
        "dtype": "float32",
        "prefix_token_id": eos,
        "tokenizer_sha256": digest,
        "prompts": PROMPTS,
        "checkpoints": [],
    }
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for step in STEPS:
        path = root / f"step-{step:09d}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["settings"]["tokenizer_sha256"] != digest:
            raise ValueError("Checkpoint/tokenizer mismatch")
        model = GPT(ModelConfig(**saved["model_config"])).cuda().eval()
        model.load_state_dict(saved["model"], strict=True)
        result = {
            "checkpoint": str(path),
            "step": step,
            "training_tokens": saved["tokens"],
            "samples": [],
        }
        del saved
        for name, prompt in PROMPTS.items():
            prompt_ids = [eos] + tokenizer.encode(prompt, add_special_tokens=False)
            ids = torch.tensor([prompt_ids], device="cuda")
            generated = []
            finish = "length"
            with torch.inference_mode():
                for _ in range(report["max_new_tokens"]):
                    if ids.shape[1] >= model.config.block_size:
                        finish = "context_limit"
                        break
                    token = model(ids)[0, -1].argmax().item()
                    generated.append(token)
                    if token == eos:
                        finish = "eos"
                        break
                    ids = torch.cat(
                        [ids, torch.tensor([[token]], device="cuda")], dim=1
                    )
            result["samples"].append(
                {
                    "name": name,
                    "prompt_ids": prompt_ids,
                    "continuation": tokenizer.decode(
                        generated,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "generated_ids": generated,
                    "finish_reason": finish,
                }
            )
        report["checkpoints"].append(result)
        (output / "generations.json").write_text(json.dumps(report, indent=2) + "\n")
        artifacts.commit()
        print(
            json.dumps(
                {"step": step, "tokens": result["training_tokens"], "completed": True}
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
    return report


@app.local_entrypoint()
def main(output_id: str):
    report = worker.remote(output_id)
    Path("distillation/results", output_id + ".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
