"""The subword GPT family (model_100m, frozen subword tokenizer) for trainer.loop.

    torchrun --nproc_per_node=<gpus> -m trainer.subword --config <config.json> \\
        --manifest <snapshot manifest> --output <run dir> [--resume <checkpoint>] \\
        [--max-seconds N] [--max-steps N] [--stop-tokens N] [--stop-file F] [--volume-name V]

Settings are a `distributed.config.DistributedConfig`. Data comes from the snapshot
manifest (`training_data.mixture.open_mixture`, optionally prefetched on a producer
thread); the token budget counts loss targets. The objective is the global mean
cross-entropy over loss targets. Evaluation is the config's held-out set: derived
windows of each source's validation split, or the manifest's frozen file; the
weighted loss uses the mixture's source weights.
"""

import argparse
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from itertools import islice
from pathlib import Path

from training_data.growing import STALL_SECONDS, STATE_ENV, DataDeadline
from training_data.mixture import check_capacity, open_mixture
from training_data.subword import BOS_TOKEN_ID, EOS_TOKEN_ID, LOSS_IGNORE_INDEX, VOCAB_SIZE

from distributed.config import DistributedConfig, DistributedRuntime, wsd_branch
from trainer import loop


def source_hashes():
    """Source identity of a subword run: the loop, optimizer, family, model and loaders."""
    return loop.source_hashes({**loop.trainer_sources(), **_family_sources()})


def _family_sources():
    import model_100m.model
    import training_data.heldout
    import training_data.live_stream
    import training_data.mixture
    import training_data.prefetch
    import training_data.prepared_reference
    import training_data.raw_token_stream
    import training_data.subword
    import training_data.token_shards
    import training_data.token_stream

    import distributed.config

    return {
        "family": Path(__file__),
        "config": Path(distributed.config.__file__),
        "model": Path(model_100m.model.__file__),
        **{name: Path(path) for name, path in model_100m.model.MODEL_DEPENDENCIES.items()},
        "loader": Path(training_data.token_stream.__file__),
        "loader_selection": Path(training_data.mixture.__file__),
        "raw_loader": Path(training_data.raw_token_stream.__file__),
        "heldout_loader": Path(training_data.heldout.__file__),
        "prefetch": Path(training_data.prefetch.__file__),
        "prepared_reference": Path(training_data.prepared_reference.__file__),
        "live_loader": Path(training_data.live_stream.__file__),
        "token_shards": Path(training_data.token_shards.__file__),
        "tokenizer": Path(training_data.subword.__file__),
    }


def validation_blocks(data, windows_per_source):
    heldout = {}
    for name in data.sources:
        blocks = data.blocks(name)
        try:
            heldout[name] = list(islice(blocks, windows_per_source))
        finally:
            blocks.close()
        if len(heldout[name]) != windows_per_source:
            raise ValueError(
                f"Source {name} has fewer than {windows_per_source} validation windows"
            )
    return heldout


def heldout_blocks(config, manifest, validation):
    """The evaluation set `config.heldout` names; never a silent fallback."""
    frozen = "heldout" in json.loads(Path(manifest).read_text())
    if config.heldout == "frozen":
        if not frozen:
            raise ValueError("heldout is frozen but the snapshot manifest has no heldout entry")
        from training_data.heldout import load_heldout

        return load_heldout(manifest, config.context)
    if frozen:
        raise ValueError("The snapshot ships a frozen held-out set; set heldout to frozen")
    # Diagnostics derived from our validation split, not capability evaluation.
    return validation_blocks(validation, config.validation_windows_per_source)


def evaluate_sources(model, heldout, *, device="cuda", bf16=True):
    """Mean loss per valid target by source and by stratum, for unequal-length windows.

    A block's `stratum` defaults to its source.
    """
    import torch

    from trainer.loss import cross_entropy

    losses, counts, stratum_sums, stratum_counts = {}, {}, {}, {}
    with torch.no_grad():
        for name, blocks in heldout.items():
            summed_loss = loss_tokens = 0
            for block in blocks:
                x = torch.tensor([block["input_ids"]], device=device)
                y = torch.tensor([block["labels"]], device=device)
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=bf16):
                    loss = cross_entropy(
                        model(x).flatten(0, 1),
                        y.flatten(),
                        reduction="sum",
                        ignore_index=LOSS_IGNORE_INDEX,
                    )
                summed_loss += float(loss)
                loss_tokens += block["loss_tokens"]
                stratum = block.get("stratum", name)
                stratum_sums[stratum] = stratum_sums.get(stratum, 0.0) + float(loss)
                stratum_counts[stratum] = stratum_counts.get(stratum, 0) + block["loss_tokens"]
            losses[name] = summed_loss / loss_tokens
            counts[name] = loss_tokens
    return {
        "per_source": losses,
        "target_counts": counts,
        "per_stratum": {k: total / stratum_counts[k] for k, total in stratum_sums.items()},
        "stratum_target_counts": stratum_counts,
        "windows": sum(len(blocks) for blocks in heldout.values()),
    }


def crop_trailing_padding(x, y):
    """Keep every supervised position, including interior masks and its causal prefix."""
    valid_columns = (y != LOSS_IGNORE_INDEX).any(dim=0).nonzero()
    # An empty rank must still participate in DDP backward with zero loss.
    length = int(valid_columns[-1]) + 1 if valid_columns.numel() else 1
    return x[:, :length].contiguous(), y[:, :length].contiguous()


def summed_loss(model, x, y, *, crop_padding=False, loss_chunk_size=0):
    """Summed cross-entropy over the valid targets of one microbatch."""
    from trainer.loss import cross_entropy

    if crop_padding:
        x, y = crop_trailing_padding(x, y)
    if loss_chunk_size:
        return model(x, labels=y, loss_chunk_size=loss_chunk_size, loss_fn=cross_entropy)
    return cross_entropy(
        model(x).flatten(0, 1), y.flatten(), reduction="sum", ignore_index=LOSS_IGNORE_INDEX
    )


def backward_batch(
    ddp, x, y, count, microbatch, world, *, bf16=True, crop_padding=False, loss_chunk_size=0
):
    """Accumulate the gradient of the global mean over `count` loss targets; returns the
    summed loss. The loop does the same through `loop.accumulate` and `SubwordFamily.loss`.
    """
    import torch

    total = torch.zeros((), device=x.device)
    parts = list(zip(x.split(microbatch), y.split(microbatch), strict=True))
    for index, (mx, my) in enumerate(parts):
        with nullcontext() if index == len(parts) - 1 else ddp.no_sync():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss = summed_loss(
                    ddp, mx, my, crop_padding=crop_padding, loss_chunk_size=loss_chunk_size
                )
            total += loss.detach()
            (loss * world / count).backward()
    return total


class SubwordData:
    """The snapshot mixture (optionally prefetched) with its validation split."""

    def __init__(self, config, manifest, deadline, state):
        import torch

        self.config, self.manifest = config, Path(manifest)

        def make():
            mixture = open_mixture(
                self.manifest,
                loader=config.data_loader,
                context=config.context,
                allow_unreviewed=config.allow_unreviewed_data,
            )
            if hasattr(mixture, "copies"):
                mixture.copies.deadline = deadline
            return mixture

        if config.prefetch_batches:
            from training_data.prefetch import PrefetchedBatches

            self.mixture = PrefetchedBatches(
                make,
                config.global_windows,
                depth=config.prefetch_batches,
                state=state,
                device=torch.device("cuda", 0),
            )
        else:
            self.mixture = make()
            if state is not None:
                self.mixture.load_state_dict(state)
        if not config.pilot and config.data_loader != "live":
            check_capacity(self.mixture, config.target_tokens)
        self.validation = open_mixture(
            self.manifest,
            loader=config.data_loader,
            context=config.context,
            split="validation",
            allow_unreviewed=config.allow_unreviewed_data,
        )
        self.identity = self.mixture.identity
        self.sources = getattr(self.mixture, "sources", None)

    def next_batch(self, rows):
        try:
            x, y = self.mixture.next_batch(rows, "cuda")
        except DataDeadline as exc:
            raise TimeoutError(str(exc)) from exc
        return x, y, None, int((y != LOSS_IGNORE_INDEX).sum())

    def state_dict(self):
        return self.mixture.state_dict()

    def load_state_dict(self, state):
        self.mixture.load_state_dict(state)

    @property
    def exposures(self):
        return self.mixture.exposures

    def close(self):
        self.mixture.close()
        if self.validation is not None:
            self.validation.close()


class SubwordFamily:
    def __init__(self, config, manifest):
        self.config, self.manifest = config, Path(manifest)
        self.settings = config.as_dict()
        self.global_rows, self.microbatch = config.global_windows, config.microbatch
        self.target_tokens, self.gradient_clip = config.target_tokens, config.gradient_clip_norm
        self.seed, self.log_every = config.seed, config.log_every
        self.evaluate_every, self.checkpoint_every = config.evaluate_every, config.checkpoint_every
        self.checkpoint_seconds, self.keep_checkpoints = 0, 0
        # Rank zero may wait for a growing database while peers wait in a broadcast.
        self.collective_timeout_seconds = STALL_SECONDS + 300 if STATE_ENV in os.environ else 300
        if config.data_loader == "live" and (
            not config.prefetch_batches or not config.allow_unreviewed_data
        ):
            raise ValueError("Live upstream data requires prefetch and allow_unreviewed_data")

    def model_config(self):
        from model_100m.model import ModelConfig

        config = self.config
        return ModelConfig(
            vocab_size=VOCAB_SIZE,
            block_size=config.context,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            eos_id=EOS_TOKEN_ID,
        )

    def learning_rate_factor(self, tokens):
        return self.config.learning_rate(tokens) / self.config.peak_lr

    def build(self, device):
        from model_100m.model import GPT

        model = GPT(self.model_config()).to(device)
        if self.config.cache_rope:
            for block in model.blocks:
                block.attention.cache_rotary(self.config.context)
        return model

    def initialize(self, model):
        """Fresh runs keep the model's own initialization."""

    def compile(self, model):
        # Compile in place so state-dict keys and optimizer parameter identity stay stable.
        if self.config.compile_model:
            model.compile(dynamic=True)
        return model

    def make_optimizer(self, model):
        from trainer.optimizers import make_optimizer

        return make_optimizer(model, self.config)

    def open_data(self, deadline, state):
        return SubwordData(self.config, self.manifest, deadline, state)

    def metadata(self):
        from model_100m.model import ModelConfig

        return {
            "architecture": {
                "module": ModelConfig.__module__,
                "config": asdict(self.model_config()),
            },
            "special_tokens": {"bos": BOS_TOKEN_ID, "eos": EOS_TOKEN_ID},
        }

    def source_files(self):
        return _family_sources()

    def branchable(self, saved):
        return wsd_branch(saved, self.config)

    def check_resume(self, saved):
        if sum(saved["data"]["loss_token_exposures"].values()) != saved["tokens"]:
            raise ValueError("Checkpoint token accounting differs from data state")

    def heldout(self, data):
        blocks = heldout_blocks(self.config, self.manifest, data.validation)
        data.validation.close()
        data.validation = None
        return {"blocks": blocks, "weights": data.sources}

    def evaluate(self, model, heldout, progress, final):
        model.eval()
        try:
            stats = evaluate_sources(model, heldout["blocks"])
        finally:
            model.train()
        losses, weights = stats["per_source"], heldout["weights"]
        return {
            **stats,
            "weighted_loss": sum(losses[n] * weights[n]["weight_percent"] / 100 for n in losses),
        }

    def prepare(self, inputs, labels):
        return {}, {"targets": (labels != LOSS_IGNORE_INDEX).sum()}

    def loss(self, model, inputs, labels, extras, counts, progress):
        loss = summed_loss(
            model,
            inputs,
            labels,
            crop_padding=self.config.crop_padding,
            loss_chunk_size=self.config.loss_chunk_size,
        )
        return loss / counts["targets"], {"loss": loss}

    def log(self, sums, counts):
        return {
            "loss": float(sums["loss"] / counts["targets"]),
            "loss_tokens": int(counts["targets"]),
        }

    def step_metrics(self, inputs, seconds, budget, world):
        return {}

    def after_step(self, data, model, progress):
        if self.config.prefetch_batches:
            data.mixture.acknowledge()

    def finish(self, data, model, progress, reason):
        config = self.config
        return {
            "loss_token_exposures": data.exposures,
            "validation_note": (
                "frozen stratified held-out set from the snapshot manifest"
                if config.heldout == "frozen"
                else f"{config.validation_windows_per_source} fixed windows/source; diagnostic only"
            ),
            "prefetch": data.mixture.stats() if config.prefetch_batches else None,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-seconds", type=int, default=3000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-tokens", type=int, help="Stop at this many targets")
    parser.add_argument("--stop-file", help="Checkpoint and exit once this file exists")
    parser.add_argument("--volume-name")
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 82800:
        raise ValueError("Session length must be between 1 and 82800 seconds")
    import time

    stop_unix = time.time() + args.max_seconds
    runtime = DistributedRuntime.from_torchrun(os.environ)
    commit = None
    if args.volume_name:
        import modal

        commit = modal.Volume.from_name(args.volume_name).commit
    config = DistributedConfig(**json.loads(Path(args.config).read_text()))
    if runtime.world_size != config.world_size:
        raise ValueError("torchrun world size differs from configuration")
    loop.train(
        SubwordFamily(config, args.manifest),
        args.output,
        runtime=runtime,
        resume=args.resume,
        stop_unix=stop_unix,
        max_steps=args.max_steps,
        stop_tokens=args.stop_tokens,
        stop_file=args.stop_file,
        commit=commit,
    )


if __name__ == "__main__":
    main()
