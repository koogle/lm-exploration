"""Sample text from a trusted checkpoint: uv run lm-generate --checkpoint <path>.

Loads the model configuration and character tokenizer stored in the checkpoint,
so no training settings are needed. Prompt characters outside the training
vocabulary are errors. Output streams one token at a time as it is sampled.
Sampling uses the full softmax distribution.
"""

import argparse

import torch
from model_100m.model import GPT, ModelConfig
from training_data.loading import CharacterTokenizer

from run_utils import load_checkpoint
from train import sample_tokens


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="trusted checkpoint.pt to load")
    parser.add_argument("--prompt", default="", help="text to continue (default: empty)")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, help="sampling seed (default: training seed + 2)")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.max_new_tokens < 0:
            raise ValueError("--max-new-tokens must be nonnegative")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available")
        checkpoint = load_checkpoint(args.checkpoint)
        model = GPT(ModelConfig(**checkpoint["config"]["model"]))
        model.load_state_dict(checkpoint["model"])
        model.to(args.device)
        tokenizer = CharacterTokenizer(checkpoint["tokenizer"]["characters"])
        seed = args.seed if args.seed is not None else checkpoint["config"]["training"]["seed"] + 2
        # Print the prompt, then each sampled character the moment it is generated.
        print(args.prompt, end="", flush=True)
        for token in sample_tokens(model, tokenizer, args.prompt, args.max_new_tokens, seed):
            print(tokenizer.decode([token]), end="", flush=True)
        print()
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
