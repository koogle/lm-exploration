"""Generate text with the standalone Simple LLM base model."""

import argparse
import json
from pathlib import Path

import torch
from model_100m.model import GPT, ModelConfig
from safetensors.torch import load_file
from tokenizers import Tokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--prompt", default="The solar system consists of")
p.add_argument("--tokens", type=int, default=128)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--temperature", type=float, default=0.8, help="0 selects greedy decoding")
p.add_argument("--top-k", type=int, default=50, help="0 samples the full vocabulary")
p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
a = p.parse_args()
if a.tokens < 0 or a.temperature < 0 or a.top_k < 0:
    p.error("Generation settings must be nonnegative")
root = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.manual_seed(a.seed)
model = GPT(ModelConfig(**json.loads((root / "config.json").read_text()))).to(a.device).eval()
model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
tokenizer = Tokenizer.from_file(str(root / "tokenizer/tokenizer.json"))
ids = [1] + tokenizer.encode(a.prompt, add_special_tokens=False).ids
with torch.inference_mode():
    for _ in range(a.tokens):
        logits = model(torch.tensor([ids[-model.config.block_size :]], device=a.device))[
            0, -1
        ].float()
        if a.temperature == 0:
            token = int(logits.argmax())
        elif a.top_k:
            values, indices = torch.topk(logits / a.temperature, min(a.top_k, logits.numel()))
            token = int(indices[torch.multinomial(values.softmax(-1), 1)])
        else:
            token = int(torch.multinomial((logits / a.temperature).softmax(-1), 1))
        ids.append(token)
        if token == 2:
            break
print(tokenizer.decode(ids, skip_special_tokens=True))
