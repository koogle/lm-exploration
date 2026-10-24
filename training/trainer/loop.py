"""Single-host CUDA DDP subword training with durable optimizer-boundary resume."""

import argparse
import hashlib
import json
import os
import random
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from itertools import islice
from pathlib import Path

from training_data.growing import STALL_SECONDS, STATE_ENV
from training_data.mixture import check_capacity, open_mixture
from training_data.subword import BOS_TOKEN_ID, EOS_TOKEN_ID, LOSS_IGNORE_INDEX, VOCAB_SIZE

from distributed.config import (
    CHECKPOINT_FORMAT,
    DistributedConfig,
    DistributedRuntime,
    validate_resume,
)


def source_hashes():
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
    import trainer.optimizers

    paths = {
        "trainer": Path(__file__),
        "config": Path(distributed.config.__file__),
        "optimizer": Path(trainer.optimizers.__file__),
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
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


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


def make_optimizer(model, config, *, fused=True):
    from trainer.optimizers import make_optimizer as create_optimizer

    return create_optimizer(model, config, fused=fused)


def evaluate_sources(model, heldout, *, device="cuda", bf16=True):
    """Mean loss per valid target by source and by stratum, for unequal-length windows.

    A block's `stratum` defaults to its source.
    """
    import torch
    from torch.nn import functional as F

    losses, counts, stratum_sums, stratum_counts = {}, {}, {}, {}
    with torch.no_grad():
        for name, blocks in heldout.items():
            summed_loss = loss_tokens = 0
            for block in blocks:
                x = torch.tensor([block["input_ids"]], device=device)
                y = torch.tensor([block["labels"]], device=device)
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=bf16):
                    loss = F.cross_entropy(
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


def backward_batch(
    ddp, x, y, count, microbatch, world, *, bf16=True, crop_padding=False, loss_chunk_size=0
):
    """Accumulate the gradient of the global mean over nonpadding loss tokens.

    DDP's autograd hooks reduce gradients during the final microbatch backward.
    """
    import torch
    from torch.nn import functional as F

    total_loss = torch.zeros((), device=x.device)
    parts = list(zip(x.split(microbatch), y.split(microbatch)))
    for index, (mx, my) in enumerate(parts):
        if crop_padding:
            mx, my = crop_trailing_padding(mx, my)
        is_last_microbatch = index == len(parts) - 1
        # no_sync must cover forward and backward for earlier microbatches.
        with nullcontext() if is_last_microbatch else ddp.no_sync():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                if loss_chunk_size:
                    loss = ddp(mx, labels=my, loss_chunk_size=loss_chunk_size)
                else:
                    loss = F.cross_entropy(
                        ddp(mx).flatten(0, 1),
                        my.flatten(),
                        reduction="sum",
                        ignore_index=LOSS_IGNORE_INDEX,
                    )
            total_loss += loss.detach()
            # DDP averages across ranks, while count covers valid tokens globally:
            # (1 / world) * sum_r grad(loss_r * world / count)
            #     = grad(sum_r loss_r / count).
            # On the final microbatch, DDP hooks reduce gradient buckets via NCCL
            # as backward runs. After backward returns, gradients are synchronized.
            (loss * world / count).backward()
    return total_loss


def train(
    config,
    manifest,
    output,
    *,
    runtime: DistributedRuntime,
    resume=None,
    max_seconds=3000,
    max_steps=None,
    stop_tokens=None,
    commit=None,
):
    """Train until `config.target_tokens`; `stop_tokens` ends the session earlier.

    `stop_tokens` is an evaluation boundary: the schedule still runs to the target, so a
    later resume continues the same run.
    """
    import numpy as np
    import torch
    import torch.distributed as dist
    from model_100m.model import GPT, ModelConfig
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not 1 <= max_seconds <= 82800 or (max_steps is not None and max_steps < 1):
        raise ValueError("Session limits must be positive; maximum session length is 82800 seconds")
    if stop_tokens is not None and not 0 < stop_tokens <= config.target_tokens:
        raise ValueError("stop_tokens must be within the token target")
    rank, world = runtime.rank, runtime.world_size
    if world != config.world_size:
        raise ValueError("torchrun world size differs from configuration")
    torch.cuda.set_device(runtime.local_rank)
    torch.set_num_threads(4)
    dist.init_process_group(
        "nccl",
        init_method=runtime.init_method,
        rank=rank,
        world_size=world,
        # Rank zero may wait for a growing database while peers wait in a broadcast.
        timeout=timedelta(seconds=STALL_SECONDS + 300 if STATE_ENV in os.environ else 300),
    )
    output, manifest = Path(output), Path(manifest)
    if config.data_loader == "live" and (
        not config.prefetch_batches or not config.allow_unreviewed_data
    ):
        raise ValueError("Live upstream data requires prefetch and allow_unreviewed_data")
    commit = commit or (lambda: None)
    stopped = False
    old_handlers = {}

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, stop)

    def primary(function):
        """Broadcast rank-zero failures before peers enter the next collective."""
        result = [None]
        if rank == 0:
            try:
                result[0] = {"value": function()}
            except Exception as exc:  # noqa: BLE001 -- propagate to every rank
                result[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(result, src=0)
        if "error" in result[0]:
            raise RuntimeError(result[0]["error"])
        return result[0]["value"]

    data = validation = None
    try:
        started = time.monotonic()
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

        def prepare():
            nonlocal data, validation
            if (output / "config.json").exists():
                raise ValueError("Use a fresh run ID, including for resume")
            output.mkdir(parents=True, exist_ok=True)

            def make_data():
                return open_mixture(
                    manifest,
                    loader=config.data_loader,
                    context=config.context,
                    allow_unreviewed=config.allow_unreviewed_data,
                )

            if config.prefetch_batches:
                from training_data.prefetch import PrefetchedBatches

                initial = None
                if resume:
                    initial = torch.load(resume, map_location="cpu", weights_only=False)["data"]
                data = PrefetchedBatches(
                    make_data,
                    config.global_windows,
                    depth=config.prefetch_batches,
                    state=initial,
                    device=torch.device("cuda", rank),
                )
            else:
                data = make_data()
            if not config.pilot and config.data_loader != "live":
                check_capacity(data, config.target_tokens)
            validation = open_mixture(
                manifest,
                loader=config.data_loader,
                context=config.context,
                split="validation",
                allow_unreviewed=config.allow_unreviewed_data,
            )
            return data.identity

        identity = primary(prepare)
        model_config = ModelConfig(
            vocab_size=VOCAB_SIZE,
            block_size=config.context,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
        )
        model = GPT(model_config).cuda()
        if config.cache_rope:
            for block in model.blocks:
                block.attention.cache_rotary(config.context)
        # Compile in place so state-dict keys and optimizer parameter identity stay stable.
        if config.compile_model:
            model.compile(dynamic=True)
        ddp = DDP(
            model,
            device_ids=[runtime.local_rank],
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
        from trainer.optimizers import set_learning_rate

        optimizer = make_optimizer(model, config)
        metadata = {
            "format": CHECKPOINT_FORMAT,
            "config": config.as_dict(),
            "architecture": {"module": ModelConfig.__module__, "config": asdict(model_config)},
            "special_tokens": {"bos": BOS_TOKEN_ID, "eos": EOS_TOKEN_ID},
            "data_identity": identity,
            "source_sha256": source_hashes(),
            "torch_version": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "resume_from": str(resume) if resume else None,
        }
        step = tokens = 0
        if resume:
            saved = None
            error = None
            try:
                # Only trusted checkpoints from our own persistent run volume.
                saved = torch.load(resume, map_location="cpu", weights_only=False)
                validate_resume(
                    saved, config, identity, metadata["source_sha256"], str(torch.__version__)
                )
                if "lineage" in saved:
                    # A continued-pretraining phase keeps its parent lineage in every checkpoint.
                    metadata["lineage"] = saved["lineage"]
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                step, tokens = saved["step"], saved["tokens"]
                if sum(saved["data"]["loss_token_exposures"].values()) != tokens:
                    raise ValueError("Checkpoint token accounting differs from data state")
                if (
                    not 0 <= tokens <= config.target_tokens
                    or step < 0
                    or len(saved["rng"]) != world
                ):
                    raise ValueError("Invalid checkpoint counters or rank RNG state")
                state = saved["rng"][rank]
                random.setstate(state["python"])
                np.random.set_state(state["numpy"])
                torch.set_rng_state(state["torch"])
                torch.cuda.set_rng_state(state["cuda"])
                if rank == 0 and not config.prefetch_batches:
                    data.load_state_dict(saved["data"])
            except Exception as exc:  # noqa: BLE001 -- every rank must fail together
                error = f"{type(exc).__name__}: {exc}"
            errors = [None] * world
            dist.all_gather_object(errors, error)
            if any(errors):
                raise ValueError(f"Resume failed: {errors}")
            del saved
        primary(lambda: atomic_json(output / "config.json", metadata))
        start_step, start_tokens = step, tokens
        last_saved = None
        final_checkpoint = None
        heldout = None
        if rank == 0:
            heldout = primary(lambda: heldout_blocks(config, manifest, validation))
        else:
            primary(lambda: None)

        if rank == 0 and validation is not None and hasattr(validation, "close"):
            validation.close()
            validation = None

        @torch.no_grad()
        def evaluate():
            model.eval()
            try:
                stats = evaluate_sources(model, heldout)
            finally:
                model.train()
            losses = stats["per_source"]
            return {
                "step": step,
                "tokens": tokens,
                **stats,
                "weighted_loss": sum(
                    losses[n] * data.sources[n]["weight_percent"] / 100 for n in losses
                ),
            }

        def evaluate_and_record():
            result = evaluate()
            with (output / "validation.jsonl").open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            return result

        def checkpoint():
            nonlocal last_saved, final_checkpoint
            rng = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            }
            states = [None] * world
            dist.all_gather_object(states, rng)

            def save():
                path = output / f"step-{step:09d}.pt"
                if path.exists():
                    raise ValueError("Refuse checkpoint overwrite")
                payload = dict(
                    metadata,
                    step=step,
                    tokens=tokens,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    data=data.state_dict(),
                    rng=states,
                )
                temporary = path.with_suffix(".pt.partial")
                torch.save(payload, temporary)
                os.replace(temporary, path)
                atomic_json(
                    output / "latest.json",
                    {"checkpoint": path.name, "step": step, "tokens": tokens},
                )
                commit()
                return str(path)

            final_checkpoint = primary(save)
            last_saved = step

        initial_validation = primary(evaluate_and_record)
        local_windows = config.global_windows // world
        reason = "token_limit"
        data_error = None
        while tokens < config.target_tokens:
            if stop_tokens is not None and tokens >= stop_tokens:
                reason = "evaluation_boundary"
                break
            flag = torch.tensor(int(stopped), device="cuda")
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if int(flag):
                reason = "signal"
                break
            elapsed = time.monotonic() - started
            limit = primary(
                lambda elapsed=elapsed, step=step: (
                    "time_limit"
                    if elapsed >= max_seconds
                    else (
                        "step_limit"
                        if max_steps is not None and step - start_step >= max_steps
                        else None
                    )
                )
            )
            if limit:
                reason = limit
                break
            full_x = full_y = None

            def fetch(tokens=tokens):
                nonlocal full_x, full_y
                cursor = data.state_dict()
                try:
                    full_x, full_y = data.next_batch(config.global_windows, "cuda")
                except StopIteration:
                    return {"stop": "source_two_pass_limit"}
                count = int((full_y != LOSS_IGNORE_INDEX).sum())
                if tokens + count > config.target_tokens:
                    # Do not consume untrained data or exceed the token ceiling.
                    data.load_state_dict(cursor)
                    return {"stop": "token_limit"}
                return {"count": count}

            torch.cuda.synchronize()
            tick = time.monotonic()
            fetch_tick = time.monotonic()
            try:
                batch = primary(fetch)
            except RuntimeError as exc:
                # All ranks received the same producer failure. Save the last
                # completed optimizer boundary, without acknowledging bad data.
                reason = "data_error"
                data_error = str(exc)
                break
            data_fetch_seconds = time.monotonic() - fetch_tick
            if "stop" in batch:
                reason = batch["stop"]
                break
            count = batch["count"]
            x = torch.empty((local_windows, config.context), dtype=torch.long, device="cuda")
            y = torch.empty_like(x)
            dist.scatter(x, list(full_x.chunk(world)) if rank == 0 else None, src=0)
            dist.scatter(y, list(full_y.chunk(world)) if rank == 0 else None, src=0)
            lr = config.learning_rate(tokens + count)
            set_learning_rate(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            compute_tick = time.monotonic()
            total_loss = backward_batch(
                ddp,
                x,
                y,
                count,
                config.microbatch,
                world,
                crop_padding=config.crop_padding,
                loss_chunk_size=config.loss_chunk_size,
            )
            # DDP has synchronized gradients; replicas now clip and update alike.
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True, foreach=True
            )
            optimizer.step()
            torch.cuda.synchronize()
            rank_stats = [None] * world
            dist.all_gather_object(
                rank_stats,
                {
                    "compute_seconds": time.monotonic() - compute_tick,
                    "nonpadding_tokens": int((y != LOSS_IGNORE_INDEX).sum()),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
            )
            # This reduction sums detached losses for logging, not gradients.
            dist.all_reduce(total_loss)
            # Wait for this GPU's queued work to finish for accurate timing.
            torch.cuda.synchronize()
            if config.prefetch_batches and rank == 0:
                data.acknowledge()
            step += 1
            tokens += count
            duration = time.monotonic() - tick

            row = {
                "step": step,
                "tokens": tokens,
                "loss_tokens": count,
                "loss": float(total_loss) / count,
                "lr": lr,
                "optimizer_lrs": [g["lr"] for g in optimizer.param_groups],
                "gradient_norm": float(norm),
                "seconds": duration,
                "nonpadding_tokens_per_second": count / duration,
                "ranks": rank_stats,
                "data_fetch_seconds": data_fetch_seconds,
                "prefetch": data.stats() if rank == 0 and config.prefetch_batches else None,
            }

            def record(row=row):
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                if row["step"] == start_step + 1 or row["step"] % config.log_every == 0:
                    print(json.dumps(row), flush=True)

            primary(record)
            if step % config.evaluate_every == 0:
                primary(evaluate_and_record)
            if step % config.checkpoint_every == 0:
                checkpoint()
        final_validation = primary(evaluate_and_record)
        if last_saved != step:
            checkpoint()

        def finish():
            report = dict(
                metadata,
                steps=step,
                session_steps=step - start_step,
                total_loss_tokens=tokens,
                lifetime_loss_tokens=metadata.get("lineage", {}).get("parent_tokens", 0) + tokens,
                session_loss_tokens=tokens - start_tokens,
                stop_reason=reason,
                data_error=data_error,
                elapsed_seconds=time.monotonic() - started,
                final_checkpoint=final_checkpoint,
                initial_validation=initial_validation,
                final_validation=final_validation,
                loss_token_exposures=data.exposures,
                validation_note=(
                    "frozen stratified held-out set from the snapshot manifest"
                    if config.heldout == "frozen"
                    else f"{config.validation_windows_per_source} fixed windows/source; "
                    "diagnostic only"
                ),
                prefetch=data.stats() if config.prefetch_batches else None,
            )
            atomic_json(output / "report.json", report)
            commit()
            return report

        return primary(finish)
    finally:
        for mixture in (data, validation):
            if mixture is not None:
                mixture.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-seconds", type=int, default=3000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-tokens", type=int, help="Stop at this many targets")
    parser.add_argument("--volume-name")
    args = parser.parse_args()
    runtime = DistributedRuntime.from_torchrun(os.environ)
    commit = None
    if args.volume_name:
        import modal

        commit = modal.Volume.from_name(args.volume_name).commit
    train(
        DistributedConfig(**json.loads(Path(args.config).read_text())),
        args.manifest,
        args.output,
        runtime=runtime,
        resume=args.resume,
        max_seconds=args.max_seconds,
        max_steps=args.max_steps,
        stop_tokens=args.stop_tokens,
        commit=commit,
    )
