"""Standalone text generation for exported custom PyTorch model bundles."""

import argparse
import json
import math
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The solar system consists of")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 selects greedy decoding")
    parser.add_argument("--top-k", type=int, default=50, help="0 samples the full vocabulary")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    if (
        args.tokens < 0
        or not math.isfinite(args.temperature)
        or args.temperature < 0
        or args.top_k < 0
    ):
        parser.error("Generation settings must be finite and nonnegative")

    # Preserve the snapshot directory when HF stores this script as a blob symlink.
    root = Path(__file__).absolute().parent
    sys.path.insert(0, str(root))
    import torch
    from model_100m.model import GPT, ModelConfig
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    model = (
        GPT(ModelConfig(**json.loads((root / "config.json").read_text()))).to(args.device).eval()
    )
    model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
    tokenizer = Tokenizer.from_file(str(root / "tokenizer/tokenizer.json"))
    ids = [1] + tokenizer.encode(args.prompt, add_special_tokens=False).ids
    with torch.inference_mode():
        for _ in range(args.tokens):
            logits = model(torch.tensor([ids[-model.config.block_size :]], device=args.device))[
                0, -1
            ].float()
            if args.temperature == 0:
                token = int(logits.argmax())
            elif args.top_k:
                values, indices = torch.topk(
                    logits / args.temperature, min(args.top_k, logits.numel())
                )
                token = int(indices[torch.multinomial(values.softmax(-1), 1)])
            else:
                token = int(torch.multinomial((logits / args.temperature).softmax(-1), 1))
            ids.append(token)
            if token == 2:
                break
    print(tokenizer.decode(ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
