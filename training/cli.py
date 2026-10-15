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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="optional JSON training settings (used by UI)")
    parser.add_argument("--resume", help="trusted checkpoint to continue")
    # Internal: the UI pre-creates runs/<id>/; lm-train otherwise picks runs/<uuid>/.
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
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


if __name__ == "__main__":
    main()
