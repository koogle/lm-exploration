"""The training loop for every model family: single-host CUDA DDP, resumable at optimizer steps.

Rank 0 reads each global batch from the family's data and scatters one slice to every
rank; ranks accumulate gradients over microbatches (bf16 autocast, DDP), then clip
and step the optimizer with the family's learning-rate factor. Rank 0 logs a row per
step to `metrics.jsonl`, evaluates at start, every `evaluate_every` steps and at the
end (`validation.jsonl`), and writes `step-<step>.pt` checkpoints (model, optimizer,
data cursor, every rank's RNG) with `latest.json`, then `report.json`. A run resumes
from an explicit checkpoint or from its output folder's `latest.json`, only with the
same settings, data identity, source files and PyTorch version (or a change the
family accepts, such as a WSD decay branch).

A family adapter supplies the model and everything specific to it:

- settings (dict, compared on resume), `global_rows`, `microbatch`, `target_tokens`,
  `gradient_clip`, `seed`, `log_every`, `evaluate_every`, `checkpoint_every` (steps,
  0 = only at the end), `checkpoint_seconds` (0 = none), `keep_checkpoints` (0 = all),
  `collective_timeout_seconds`;
- `learning_rate_factor(tokens)`; `build(device)` → model; `initialize(model)` (fresh
  runs only); `compile(model)` → the module DDP wraps; `make_optimizer(model)`;
- `open_data(deadline, state)` (rank 0; `state` is a resumed cursor or None) → data
  with `identity`, `next_batch(rows)` → (inputs, labels, row names or None, budget
  tokens), `state_dict()`, `load_state_dict()`, `close()`; StopIteration means
  exhausted, TimeoutError late;
- `prepare(inputs, labels)` → (per-row extras, local target counts) and
  `loss(model, inputs, labels, extras, counts, progress)` → (objective normalized by the
  global counts, detached sums); `log(sums, counts)` → logged values;
- `heldout(data)` and `evaluate(model, heldout, progress, final)` (rank 0);
- `metadata()`, `source_files()`, `branchable(saved)`, `check_resume(saved)`,
  `after_step(data, model, progress)` and `finish(data, model, progress, reason)` (rank 0).
"""

import contextlib
import faulthandler
import hashlib
import json
import math
import os
import random
import signal
import time
from datetime import timedelta
from pathlib import Path

CHECKPOINT_FORMAT = "lm-exploration-ddp-v2"
HANG_DUMP_SECONDS = 1200


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def source_hashes(paths):
    """SHA-256 of each named source file."""
    return {name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for name, p in paths.items()}


def trainer_sources():
    """The loop and optimizer files every family's source identity includes."""
    import trainer.optimizers
    import trainer.roles

    return {
        "trainer": Path(__file__),
        "optimizer": Path(trainer.optimizers.__file__),
        "roles": Path(trainer.roles.__file__),
    }


def latest_checkpoint(output):
    """The checkpoint `latest.json` in `output` names, or None."""
    latest = Path(output) / "latest.json"
    if not latest.exists():
        return None
    return Path(output) / json.loads(latest.read_text())["checkpoint"]


def accumulate(ddp, parts, objective, world):
    """Backward each microbatch's objective; returns the summed detached sums.

    `parts` are microbatch argument tuples; `objective(*part)` returns (objective,
    sums). DDP averages gradients across ranks, so each objective is scaled by `world`
    and the optimizer sees the gradient of the global normalized objective. DDP hooks
    reduce gradient buckets during the final microbatch's backward.
    """
    import torch

    totals = {}
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        # no_sync must cover forward and backward for earlier microbatches.
        with contextlib.nullcontext() if last else ddp.no_sync():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value, sums = objective(*part)
            (value * world).backward()
        for key, sum_ in sums.items():
            totals[key] = totals.get(key, 0) + sum_.detach()
    return totals


def _split(value, size):
    return value.split(size) if value is not None else None


def check_resume(saved, family, identity, sources, torch_version):
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported checkpoint format")
    if saved["config"] != family.settings and not family.branchable(saved):
        raise ValueError("Resume requires identical training settings and GPU count")
    if saved["data_identity"] != identity:
        raise ValueError("Checkpoint dataset/tokenizer identity mismatch")
    if saved["source_sha256"] != sources or saved["torch_version"] != torch_version:
        raise ValueError("Resume requires identical training source and PyTorch version")


def train(
    family,
    output,
    *,
    runtime,
    resume=None,
    stop_unix=None,
    max_steps=None,
    stop_tokens=None,
    stop_file=None,
    commit=None,
    save=True,
):
    """Train until `family.target_tokens`; returns the report (every rank).

    `stop_tokens` ends the session earlier at a token count (the schedule still runs
    to the target, so a later resume continues the same run); `stop_unix`,
    `max_steps`, SIGTERM/SIGINT and an existing `stop_file` end it with a checkpoint.
    `save=False` writes no checkpoints (throughput benchmarks).
    """
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from trainer.optimizers import set_learning_rate

    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if stop_tokens is not None and not 0 < stop_tokens <= family.target_tokens:
        raise ValueError("stop_tokens must be within the token target")
    rank, world = runtime.rank, runtime.world_size
    rows = family.global_rows // world
    if rows * world != family.global_rows or rows % family.microbatch:
        raise ValueError("Global rows must divide into whole per-GPU microbatches")
    stop_unix = stop_unix if stop_unix is not None else math.inf
    torch.cuda.set_device(runtime.local_rank)
    torch.set_num_threads(4)
    dist.init_process_group(
        "nccl",
        init_method=runtime.init_method,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=family.collective_timeout_seconds),
    )
    device = torch.device("cuda", runtime.local_rank)
    output = Path(output)
    commit = commit or (lambda: None)
    resume = Path(resume) if resume else latest_checkpoint(output)
    stopped = False
    old_handlers = {}

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, stop)

    def primary(function):
        """Broadcast rank-zero results (or failures) before peers enter the next collective."""
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

    data = None
    try:
        started = time.monotonic()
        random.seed(family.seed)
        np.random.seed(family.seed)
        torch.manual_seed(family.seed)
        torch.cuda.manual_seed(family.seed)

        saved, load_error = None, None
        if resume:
            try:
                # Only trusted checkpoints from our own run volumes.
                saved = torch.load(resume, map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001 -- every rank must fail together
                load_error = f"{type(exc).__name__}: {exc}"

        def prepare():
            nonlocal data
            output.mkdir(parents=True, exist_ok=True)
            data = family.open_data(stop_unix, saved["data"] if saved else None)
            return data.identity

        identity = primary(prepare)
        model = family.build(device)
        optimizer = family.make_optimizer(model)
        sources = source_hashes({**trainer_sources(), **family.source_files()})
        metadata = {
            "format": CHECKPOINT_FORMAT,
            "config": family.settings,
            **family.metadata(),
            "data_identity": identity,
            "source_sha256": sources,
            "torch_version": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "resume_from": str(resume) if resume else None,
        }
        step = tokens = 0
        if resume:
            error = load_error
            try:
                if error:
                    raise ValueError(error)
                check_resume(saved, family, identity, sources, str(torch.__version__))
                family.check_resume(saved)
                if "lineage" in saved:
                    # A continued-pretraining phase keeps its parent lineage in every checkpoint.
                    metadata["lineage"] = saved["lineage"]
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                step, tokens = saved["step"], saved["tokens"]
                if (
                    not 0 <= tokens <= family.target_tokens
                    or step < 0
                    or len(saved["rng"]) != world
                ):
                    raise ValueError("Invalid checkpoint counters or rank RNG state")
                state = saved["rng"][rank]
                random.setstate(state["python"])
                np.random.set_state(state["numpy"])
                torch.set_rng_state(state["torch"])
                torch.cuda.set_rng_state(state["cuda"])
            except Exception as exc:  # noqa: BLE001 -- every rank must fail together
                error = f"{type(exc).__name__}: {exc}"
            errors = [None] * world
            dist.all_gather_object(errors, error)
            if any(errors):
                raise ValueError(f"Resume failed: {errors}")
            del saved
        else:
            family.initialize(model)
        primary(lambda: atomic_json(output / "config.json", metadata))
        ddp = DDP(
            family.compile(model),
            device_ids=[runtime.local_rank],
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
        start_step, start_tokens = step, tokens
        last_saved, final_checkpoint = None, None
        heldout = None

        def load_heldout():
            nonlocal heldout
            heldout = family.heldout(data)

        primary(load_heldout)

        def evaluate(final=False):
            def run():
                progress = {"step": step, "tokens": tokens}
                result = {**progress, **family.evaluate(model, heldout, progress, final)}
                with (output / "validation.jsonl").open("a") as handle:
                    handle.write(json.dumps(result) + "\n")
                return result

            return primary(run)

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
                if family.keep_checkpoints:
                    for old in sorted(output.glob("step-*.pt"))[: -family.keep_checkpoints]:
                        old.unlink()
                commit()
                return str(path)

            final_checkpoint = primary(save)
            last_saved = step

        initial_validation = evaluate()
        last_checkpoint = time.monotonic()
        reason, data_error = "token_limit", None
        while tokens < family.target_tokens:
            # A step that takes this long is stuck: print every thread's stack (repeatedly).
            faulthandler.dump_traceback_later(HANG_DUMP_SECONDS, repeat=True)
            if stop_tokens is not None and tokens >= stop_tokens:
                reason = "evaluation_boundary"
                break
            flag = torch.tensor(int(stopped), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if int(flag):
                reason = "signal"
                break

            def limits(step=step):
                if time.time() >= stop_unix:
                    return "time_limit"
                if max_steps is not None and step - start_step >= max_steps:
                    return "step_limit"
                if stop_file and os.path.exists(stop_file):
                    return "stop_requested"
                return None

            limit = primary(limits)
            if limit:
                reason = limit
                break
            full = {}

            def fetch(tokens=tokens, full=full):
                cursor = data.state_dict()
                try:
                    inputs, labels, names, budget = data.next_batch(family.global_rows)
                except StopIteration:
                    return {"stop": "data_exhausted"}
                except TimeoutError:
                    return {"stop": "data_late"}
                if tokens + budget > family.target_tokens:
                    # Do not consume untrained data or exceed the token target.
                    data.load_state_dict(cursor)
                    return {"stop": "token_limit"}
                full["inputs"] = torch.as_tensor(inputs, device=device).long()
                full["labels"] = torch.as_tensor(labels, device=device).long()
                counted = {n: names.count(n) for n in sorted(set(names))} if names else None
                return {"budget": budget, "sources": counted, "context": full["inputs"].shape[1]}

            torch.cuda.synchronize()
            tick = time.monotonic()
            try:
                batch = primary(fetch)
            except RuntimeError as exc:
                # All ranks received the same producer failure. Save the last
                # completed optimizer boundary, without acknowledging bad data.
                reason, data_error = "data_error", str(exc)
                break
            data_fetch_seconds = time.monotonic() - tick
            if "stop" in batch:
                reason = batch["stop"]
                break
            inputs = torch.empty((rows, batch["context"]), dtype=torch.long, device=device)
            labels = torch.empty_like(inputs)
            dist.scatter(inputs, list(full["inputs"].chunk(world)) if rank == 0 else None, src=0)
            dist.scatter(labels, list(full["labels"].chunk(world)) if rank == 0 else None, src=0)
            full.clear()
            extras, counts = family.prepare(inputs, labels)
            for value in counts.values():
                dist.all_reduce(value)
            progress = {"step": step, "tokens": tokens}
            factor = family.learning_rate_factor(tokens + batch["budget"])
            set_learning_rate(optimizer, factor)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            compute_tick = time.monotonic()
            parts = list(
                zip(
                    inputs.split(family.microbatch),
                    labels.split(family.microbatch),
                    *([_split(v, family.microbatch) for v in extras.values()]),
                    strict=True,
                )
            )
            keys = list(extras)

            def objective(x, y, *rest, keys=keys, counts=counts, progress=progress):
                return family.loss(ddp, x, y, dict(zip(keys, rest, strict=True)), counts, progress)

            sums = accumulate(ddp, parts, objective, world)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), family.gradient_clip, foreach=True
            )
            for value in sums.values():
                dist.all_reduce(value)
            logged = family.log(sums, counts)
            values = [float(norm), *(v for v in logged.values() if isinstance(v, float))]
            if not all(math.isfinite(v) for v in values):
                reason = "non_finite"
                break
            optimizer.step()
            torch.cuda.synchronize()
            rank_stats = [None] * world
            dist.all_gather_object(
                rank_stats,
                {
                    "compute_seconds": time.monotonic() - compute_tick,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
            )
            step += 1
            tokens += batch["budget"]
            seconds = time.monotonic() - tick
            row = {
                "step": step,
                "tokens": tokens,
                "budget_tokens": batch["budget"],
                **logged,
                "lr_factor": factor,
                "optimizer_lrs": [g["lr"] for g in optimizer.param_groups],
                "grad_norm": float(norm),
                "seconds": seconds,
                "tokens_per_second": batch["budget"] / seconds,
                "data_fetch_seconds": data_fetch_seconds,
                "sources": batch["sources"],
                "ranks": rank_stats,
            }

            def record(row=row, seconds=seconds, inputs=inputs, budget=batch["budget"]):
                row.update(family.step_metrics(inputs, seconds, budget, world))
                family.after_step(data, model, {"step": row["step"], "tokens": row["tokens"]})
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                if row["step"] == start_step + 1 or row["step"] % family.log_every == 0:
                    print(json.dumps(row), flush=True)

            primary(record)
            if step % family.evaluate_every == 0:
                evaluate()
            due = family.checkpoint_every and step % family.checkpoint_every == 0
            # Rank 0's clock decides, so every rank enters checkpoint()'s collective together.
            late = family.checkpoint_seconds and primary(
                lambda since=last_checkpoint: time.monotonic() - since >= family.checkpoint_seconds
            )
            if save and (due or late):
                checkpoint()
                last_checkpoint = time.monotonic()
        faulthandler.cancel_dump_traceback_later()
        final_validation = evaluate(final=True)
        if save and reason != "non_finite" and last_saved != step:
            checkpoint()

        def finish():
            progress = {"step": step, "tokens": tokens}
            report = dict(
                metadata,
                steps=step,
                session_steps=step - start_step,
                tokens=tokens,
                lifetime_tokens=metadata.get("lineage", {}).get("parent_tokens", 0) + tokens,
                session_tokens=tokens - start_tokens,
                stop_reason=reason,
                data_error=data_error,
                elapsed_seconds=time.monotonic() - started,
                final_checkpoint=final_checkpoint,
                initial_validation=initial_validation,
                final_validation=final_validation,
                **family.finish(data, model, progress, reason),
            )
            atomic_json(output / "report.json", report)
            commit()
            print(
                json.dumps({"report": {k: report[k] for k in ("steps", "tokens", "stop_reason")}})
            )
            return report

        return primary(finish)
    finally:
        faulthandler.cancel_dump_traceback_later()
        if data is not None:
            data.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        dist.destroy_process_group()
