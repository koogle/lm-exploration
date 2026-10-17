"""Single-host CUDA DDP subword training with durable optimizer-boundary resume."""

import argparse
import hashlib
import json
import os
import random
import signal
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

from training_data.subword import LOSS_IGNORE_INDEX

from trainer.distributed_config import DistributedConfig, validate_resume


def source_hashes():
    import model_100m.model
    import training_data.raw_token_stream
    import training_data.subword
    import training_data.token_stream

    import trainer.distributed_config

    paths = {
        "trainer": Path(__file__),
        "config": Path(trainer.distributed_config.__file__),
        "model": Path(model_100m.model.__file__),
        "loader": Path(training_data.token_stream.__file__),
        "raw_loader": Path(training_data.raw_token_stream.__file__),
        "tokenizer": Path(training_data.subword.__file__),
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def backward_batch(ddp, x, y, count, microbatch, world, *, bf16=True):
    """Accumulate the gradient of the global mean over nonpadding loss tokens."""
    import torch
    from torch.nn import functional as F

    total_loss = torch.zeros((), device=x.device)
    parts = list(zip(x.split(microbatch), y.split(microbatch)))
    for index, (mx, my) in enumerate(parts):
        with ddp.no_sync() if index < len(parts) - 1 else nullcontext():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss = F.cross_entropy(
                    ddp(mx).flatten(0, 1),
                    my.flatten(),
                    reduction="sum",
                    ignore_index=LOSS_IGNORE_INDEX,
                )
            total_loss += loss.detach()
            # DDP averages gradients: compensate before global token normalization.
            (loss * world / count).backward()
    return total_loss


def train(config, manifest, output, *, resume=None, max_seconds=3000, max_steps=None, commit=None):
    import numpy as np
    import torch
    import torch.distributed as dist
    from model_100m.model import GPT, ModelConfig
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from training_data.token_stream import TokenMixture

    if not 1 <= max_seconds <= 82800 or (max_steps is not None and max_steps < 1):
        raise ValueError("Session limits must be positive; maximum session length is 82800 seconds")
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if world != config.world_size:
        raise ValueError("torchrun world size differs from configuration")
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    output, manifest = Path(output), Path(manifest)
    if json.loads(manifest.read_text()).get("format") == "scholarly-raw-v1":
        from training_data.raw_token_stream import RawTokenMixture

        TokenMixture = RawTokenMixture
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

    try:
        started = time.monotonic()
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        data = validation = None

        def prepare():
            nonlocal data, validation
            if (output / "config.json").exists():
                raise ValueError("Use a fresh run ID, including for resume")
            output.mkdir(parents=True, exist_ok=True)
            data = TokenMixture(
                manifest,
                context=config.context,
                allow_unreviewed=(config.pilot or config.experimental_data),
            )
            if not config.pilot:
                capacity = min(
                    2 * s["splits"]["train"]["loss_tokens"] * 100 / s["weight_percent"]
                    for s in data.sources.values()
                )
                if config.target_tokens > capacity:
                    raise ValueError("Snapshot cannot support target mixture within two passes")
            validation = TokenMixture(
                manifest,
                context=config.context,
                split="validation",
                allow_unreviewed=(config.pilot or config.experimental_data),
            )
            return data.identity

        identity = primary(prepare)
        model_config = ModelConfig(
            vocab_size=32768,
            block_size=config.context,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
        )
        model = GPT(model_config).cuda()
        ddp = DDP(model, device_ids=[rank], gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.peak_lr,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
            fused=True,
        )
        metadata = {
            "format": "scholarly-ddp-v1",
            "config": config.as_dict(),
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
                if rank == 0:
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
            # A fixed small validation set: diagnostics, not capability evaluation.
            heldout = primary(
                lambda: {name: next(validation.blocks(name)) for name in validation.sources}
            )
        else:
            primary(lambda: None)

        @torch.no_grad()
        def evaluate():
            model.eval()
            losses = {}
            try:
                for name, block in heldout.items():
                    x = torch.tensor([block["input_ids"]], device="cuda")
                    y = torch.tensor([block["labels"]], device="cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = F.cross_entropy(
                            model(x).flatten(0, 1),
                            y.flatten(),
                            reduction="sum",
                            ignore_index=LOSS_IGNORE_INDEX,
                        )
                    losses[name] = float(loss) / block["loss_tokens"]
            finally:
                model.train()
            return {
                "step": step,
                "tokens": tokens,
                "per_source": losses,
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
        while tokens < config.target_tokens:
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
            batch = primary(fetch)
            if "stop" in batch:
                reason = batch["stop"]
                break
            count = batch["count"]
            x = torch.empty((local_windows, config.context), dtype=torch.long, device="cuda")
            y = torch.empty_like(x)
            dist.scatter(x, list(full_x.chunk(world)) if rank == 0 else None, src=0)
            dist.scatter(y, list(full_y.chunk(world)) if rank == 0 else None, src=0)
            lr = config.learning_rate(tokens + count)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total_loss = backward_batch(ddp, x, y, count, config.microbatch, world)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True, foreach=True
            )
            optimizer.step()
            dist.all_reduce(total_loss)
            torch.cuda.synchronize()
            step += 1
            tokens += count
            duration = time.monotonic() - tick

            row = {
                "step": step,
                "tokens": tokens,
                "loss_tokens": count,
                "loss": float(total_loss) / count,
                "lr": lr,
                "gradient_norm": float(norm),
                "seconds": duration,
            }

            def record(row=row):
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                if row["step"] == start_step + 1 or row["step"] % 10 == 0:
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
                session_loss_tokens=tokens - start_tokens,
                stop_reason=reason,
                elapsed_seconds=time.monotonic() - started,
                final_checkpoint=final_checkpoint,
                initial_validation=initial_validation,
                final_validation=final_validation,
                loss_token_exposures=data.exposures,
                validation_note="One fixed window/source; diagnostic only",
            )
            atomic_json(output / "report.json", report)
            commit()
            return report

        return primary(finish)
    finally:
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
    parser.add_argument("--volume-name")
    args = parser.parse_args()
    commit = None
    if args.volume_name:
        import modal

        commit = modal.Volume.from_name(args.volume_name).commit
    train(
        DistributedConfig(**json.loads(Path(args.config).read_text())),
        args.manifest,
        args.output,
        resume=args.resume,
        max_seconds=args.max_seconds,
        max_steps=args.max_steps,
        commit=commit,
    )
