"""Load the Simple LLM 124M smoke checkpoint; output is expected to be incoherent."""

import argparse
import json
from pathlib import Path

import torch
from model_100m.model import GPT, ModelConfig
from safetensors.torch import load_file
from tokenizers import Tokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--prompt", default="Hello")
p.add_argument("--tokens", type=int, default=32)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
root = Path(__file__).resolve().parent
torch.manual_seed(a.seed)
torch.set_num_threads(2)
model = GPT(ModelConfig(**json.loads((root / "config.json").read_text())))
model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
model.eval()
tokenizer = Tokenizer.from_file(str(root / "tokenizer/tokenizer.json"))
ids = [1] + tokenizer.encode(a.prompt, add_special_tokens=False).ids
with torch.no_grad():
    for _ in range(a.tokens):
        logits = model(torch.tensor([ids[-model.config.block_size :]]))[:, -1]
        token = torch.multinomial(logits.softmax(-1), 1).item()
        ids.append(token)
        if token == 2:
            break
print(tokenizer.decode(ids, skip_special_tokens=True))
