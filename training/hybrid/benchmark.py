"""Training-step throughput and MFU of a hybrid run config on synthetic packed documents.

    torchrun --nproc_per_node=<gpus> -m model_hybrid.benchmark --config <run.json> \\
        --steps 20 [--microbatch 4] [--document-tokens 1000]

Runs the real training step (trainer.loop with the hybrid family: optimizer,
compile, DDP) without checkpoints or evaluation. Document lengths are geometric with the given mean, close to the
recipe's mixture (ClimbMix ~600, code ~1,700, math ~1,400 tokens). Prints one JSON
line per step from the trainer; summarize with `summarize`.
"""

import argparse
import dataclasses
import json
import os
import tempfile
import time

import numpy as np

from distributed.config import DistributedRuntime
from trainer import loop

from . import tokenizer as glm
from .train import HybridFamily, RunConfig


class SyntheticMixture:
    def __init__(self, context, vocab, document_tokens, seed=0):
        self.context, self.vocab, self.mean = context, vocab, document_tokens
        self.rng = np.random.default_rng(seed)

    def next_batch(self, rows):
        tokens = self.rng.integers(0, glm.TOKENIZER_SIZE - 64, (rows, self.context + 1))
        ends = self.rng.random((rows, self.context + 1)) < 1 / self.mean
        tokens[ends] = glm.EOS_ID
        return tokens[:, :-1], tokens[:, 1:], ["synthetic"] * rows

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        """Synthetic batches have no cursor."""


def summarize(lines, skip=5):
    """Median throughput and MFU of the steps after the first `skip` (compile, warmup)."""
    steps = [json.loads(line) for line in lines if line.startswith('{"step"')]
    timed = [s for s in steps if s["step"] > skip]
    return {
        "steps_timed": len(timed),
        "tokens_per_second": float(np.median([s["tokens_per_second"] for s in timed])),
        "mfu": float(np.median([s["mfu"] for s in timed])),
        "loss_first": steps[0]["loss"] if steps else None,
        "loss_last": steps[-1]["loss"] if steps else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--global-rows", type=int)
    parser.add_argument("--document-tokens", type=int, default=1000)
    args = parser.parse_args()
    config = RunConfig.load(args.config)
    overrides = {"recipe": "", "hf_repo": "", "log_every_steps": 1, "checkpoint_seconds": 10**9}
    if args.microbatch:
        overrides["microbatch"] = args.microbatch
    if args.global_rows:
        overrides["global_rows"] = args.global_rows
    config = dataclasses.replace(config, **overrides)
    context = config.model["block_size"]
    with tempfile.TemporaryDirectory() as run_dir:
        mixture = SyntheticMixture(context, config.model["vocab_size"], args.document_tokens)
        family = HybridFamily(config, run_dir, data=({"synthetic": True}, lambda: mixture))
        loop.train(
            family,
            run_dir,
            runtime=DistributedRuntime.from_torchrun(os.environ),
            stop_unix=time.time() + 3600,
            max_steps=args.steps,
            save=False,
        )


if __name__ == "__main__":
    main()
