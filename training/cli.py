"""Train the architecture in 100m/model.py: uv run lm-train.

Edit trainer/config.py to change training settings; model dimensions belong in
100m/model.py. --config is an optional JSON settings file used by the local UI.
--resume optionally continues a trusted checkpoint; steps is the TOTAL target.
FP32 ~100M training needs several GiB; CPU can be slow. No run starts on import.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

GENERATE_DESCRIPTION = """Sample text from a trusted checkpoint: uv run lm-generate --checkpoint <path>.

Loads the model configuration and character tokenizer stored in the checkpoint,
so no training settings are needed. Prompt characters outside the training
vocabulary are errors. Output streams one token at a time as it is sampled.
Sampling uses the full softmax distribution.
"""


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="optional JSON training settings (used by UI)")
    parser.add_argument("--resume", help="trusted checkpoint to continue")
    # Internal: the UI pre-creates runs/<id>/; lm-train otherwise picks runs/<uuid>/.
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def build_generate_parser():
    parser = argparse.ArgumentParser(description=GENERATE_DESCRIPTION)
    parser.add_argument("--checkpoint", required=True, help="trusted checkpoint file to load")
    parser.add_argument("--prompt", default="", help="text to continue (default: empty)")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, help="sampling seed (default: training seed + 2)")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    # Keep the entry point thin; --help does not import PyTorch or the trainer.
    from trainer.config import TrainConfig
    from trainer.loop import train

    try:
        config = TrainConfig()
        if options.config:
            config = TrainConfig(**json.loads(options.config.read_text(encoding="utf-8")))
        if options.resume:
            config = replace(config, resume=options.resume)
        train(config, options.run_dir)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))


def generate():
    parser = build_generate_parser()
    args = parser.parse_args()
    # Imported here for the same reason as in main().
    import torch
    from model_100m.model import GPT, ModelConfig
    from training_data.loading import CharacterTokenizer

    from trainer.checkpoints import load_checkpoint
    from trainer.sampling import sample_tokens

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
