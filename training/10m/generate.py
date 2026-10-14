"""Standalone sampling from a downloaded Hugging Face Shakespeare model folder."""

import argparse
import json
from pathlib import Path

import torch
from model import GPT, ModelConfig
from safetensors.torch import load_file


def load_model(folder, device="cpu"):
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    tokenizer = json.loads((folder / "tokenizer.json").read_text())
    model = GPT(ModelConfig(**config)).to(device).eval()
    model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
    return model, tokenizer["characters"]


@torch.no_grad()
def generate(model, characters, prompt, tokens=300, seed=1339):
    indices = {character: index for index, character in enumerate(characters)}
    if set(prompt) - indices.keys():
        raise ValueError("prompt contains characters absent from the training vocabulary")
    context = [indices[c] for c in prompt] or [0]
    random = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    output = []
    for _ in range(tokens):
        logits = model(torch.tensor([context[-model.config.block_size :]], device=device))
        probabilities = logits[0, -1].softmax(-1).cpu()
        token = torch.multinomial(probabilities, 1, generator=random).item()
        context.append(token)
        output.append(characters[token])
    return prompt + "".join(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default=".")
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1339)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()
    if args.tokens < 0:
        parser.error("tokens must be nonnegative")
    model, characters = load_model(args.folder, args.device)
    print(generate(model, characters, args.prompt, args.tokens, args.seed))
