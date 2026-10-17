"""Bounded GPU pilot; preserves token counts, data cursors and optimizer state."""

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path


def train_pilot(manifest, output, provenance, commit):
    import torch
    from model_100m.model import GPT, ModelConfig
    from torch.nn import functional as F
    from training_data.subword import TOKENIZER_SHA256, load_frozen
    from training_data.token_stream import TokenMixture

    if output.exists():
        raise ValueError("Run directory already exists; preserve previous artifacts")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    tokenizer = load_frozen()
    for folder in ("model_100m", "model_10m", "trainer", "training_data"):
        shutil.copytree(
            Path("/project") / folder,
            output / "source" / folder,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    shutil.copytree("/project/tokenizer/structured-code-v1", output / "tokenizer")
    shutil.copyfile(manifest, output / "data-manifest.json")
    config = ModelConfig(vocab_size=32768, block_size=1024)
    settings = {
        "batch_size": 2,
        "accumulation": 4,
        "max_steps": 600,
        "max_loss_tokens": 5_000_000,
        "max_seconds": 1200,
        "peak_lr": 3e-4,
        "warmup_steps": 30,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "precision": "bf16 autocast; fp32 weights",
        "seed": 42,
    }
    data = TokenMixture(manifest, context=config.block_size, allow_unreviewed=True)
    # Exercise real CUDA BF16 forward/backward on a tiny model before allocating the full one.
    tiny = GPT(ModelConfig(vocab_size=32768, block_size=32, n_layer=1, n_head=2, n_embd=64)).cuda()
    tiny_optimizer = torch.optim.AdamW(tiny.parameters(), lr=0.01)
    tiny_x = torch.tensor(
        [[1] + tokenizer.encode("def add(a, b): return a + b").ids], device="cuda"
    )
    tiny_y = torch.cat([tiny_x[:, 1:], torch.tensor([[2]], device="cuda")], dim=1)
    tiny_losses = []
    for _ in range(30):
        tiny_optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tiny_loss = F.cross_entropy(tiny(tiny_x).flatten(0, 1), tiny_y.flatten())
        tiny_loss.backward()
        tiny_optimizer.step()
        tiny_losses.append(float(tiny_loss.detach()))
    if not tiny_losses[-1] < 0.5 * tiny_losses[0]:
        raise ValueError("CUDA tiny-set fitting qualification failed")
    del tiny, tiny_optimizer, tiny_loss
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = GPT(config).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["peak_lr"], betas=tuple(settings["betas"]), weight_decay=0.1
    )
    metadata = {
        "provenance": provenance,
        "model_name": "100m",
        "model_config": asdict(config),
        "settings": settings,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "manifest_sha256": data.identity["manifest_sha256"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "training_ready": False,
        "tiny_fit_losses": [tiny_losses[0], tiny_losses[-1]],
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2))
    # Freeze one document-local window per source for repeatable pilot validation.
    validation = TokenMixture(
        manifest, context=config.block_size, split="validation", allow_unreviewed=True
    )
    heldout = {name: next(validation.blocks(name)) for name in validation.sources}

    def tensors(block):
        return (
            torch.tensor([block["input_ids"]], device="cuda"),
            torch.tensor([block["labels"]], device="cuda"),
        )

    def loss_sum(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.cross_entropy(
                model(x).flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum"
            )

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = {}
        for name, block in heldout.items():
            x, y = tensors(block)
            losses[name] = loss_sum(x, y).item() / block["loss_tokens"]
        model.train()
        return {
            "per_source": losses,
            "weighted_loss": sum(
                losses[n] * data.sources[n]["weight_percent"] / 100 for n in losses
            ),
        }

    def save(step):
        path = output / f"step-{step:06d}.pt"
        if path.exists():
            raise ValueError("Refuse checkpoint overwrite")
        started = time.monotonic()
        torch.save(
            dict(
                format="scholarly-pilot-v1",
                **metadata,
                step=step,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                data=data.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
            ),
            path.with_suffix(".pt.partial"),
        )
        path.with_suffix(".pt.partial").rename(path)
        commit()
        return path, time.monotonic() - started

    initial = evaluate()
    print(json.dumps({"initial_validation": initial, **metadata}), flush=True)
    history = []
    start = time.monotonic()
    reason = "step_limit"
    checkpoint_seconds = 0.0
    resume_verified = False
    step = 0
    compute_seconds = 0.0
    while step < settings["max_steps"]:
        exposures = sum(data.exposures.values())
        if time.monotonic() - start >= settings["max_seconds"]:
            reason = "time_limit"
            break
        if (
            exposures + settings["batch_size"] * settings["accumulation"] * config.block_size
            > settings["max_loss_tokens"]
        ):
            reason = "token_limit"
            break
        # Fetch the complete optimizer batch first; exhaustion cannot consume a partial step.
        cursor = data.state_dict()
        try:
            batches = [
                data.next_batch(settings["batch_size"], "cuda")
                for _ in range(settings["accumulation"])
            ]
        except StopIteration:
            data.load_state_dict(cursor)
            reason = "source_two_pass_limit"
            break
        count = sum(int((y != -100).sum()) for _, y in batches)
        lr = settings["peak_lr"] * min((step + 1) / settings["warmup_steps"], 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        tick = time.monotonic()
        total = 0.0
        for x, y in batches:
            loss = loss_sum(x, y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            total += loss.detach().item()
            (loss / count).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - tick
        compute_seconds += elapsed
        step += 1
        row = {
            "step": step,
            "loss": total / count,
            "loss_tokens": count,
            "lr": lr,
            "gradient_norm": float(norm),
            "compute_seconds": elapsed,
            "cumulative_tokens": sum(data.exposures.values()),
        }
        history.append(row)
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if step % 10 == 0 or step == 1:
            print(json.dumps(row), flush=True)
        if step == 10 or step % 100 == 0:
            checkpoint, duration = save(step)
            checkpoint_seconds += duration
            if step == 10:
                # Reload actual serialized weights/optimizer/RNG and compare the next batch/logits.
                saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
                state = data.state_dict()
                x1, y1 = data.next_batch(2, "cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    before = model(x1).clone()
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                torch.set_rng_state(saved["torch_rng"].cpu())
                torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
                data.load_state_dict(saved["data"])
                x2, y2 = data.next_batch(2, "cuda")
                assert torch.equal(x1, x2) and torch.equal(y1, y2)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    torch.testing.assert_close(model(x2), before, rtol=0, atol=0)
                data.load_state_dict(state)
                del saved, before, x1, y1, x2, y2
                resume_verified = True
                print("Checkpoint reload and next-batch/logit equivalence passed", flush=True)
    if not (output / f"step-{step:06d}.pt").exists():
        _, duration = save(step)
        checkpoint_seconds += duration
    final = evaluate()
    samples = {}
    model.eval()
    with torch.no_grad():
        for prompt in [
            "The purpose of a scientific experiment is",
            "def fibonacci(n):\n",
            "2 + 2 =",
        ]:
            torch.manual_seed(42)
            ids = [1] + tokenizer.encode(prompt).ids
            x = torch.tensor([ids], device="cuda")
            for _ in range(64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x[:, -config.block_size :])[:, -1].float()
                logits[:, [0, 1, 3]] = -float("inf")
                token = torch.multinomial(logits.softmax(-1), 1)
                x = torch.cat([x, token], dim=1)
                if token.item() == 2:
                    break
            samples[prompt] = tokenizer.decode(x[0].tolist())
    tokens = sum(data.exposures.values())
    report = dict(
        **metadata,
        steps=step,
        stop_reason=reason,
        resume_verified=resume_verified,
        loss_token_exposures=data.exposures,
        total_loss_tokens=tokens,
        source_accounting={
            name: {
                "unique_train_loss_tokens": source["splits"]["train"]["loss_tokens"],
                "exposures": data.exposures[name],
                "effective_passes": data.exposures[name] / source["splits"]["train"]["loss_tokens"],
                "actual_percent": 100 * data.exposures[name] / tokens if tokens else 0,
                "target_percent": source["weight_percent"],
            }
            for name, source in data.sources.items()
        },
        initial_validation=initial,
        final_validation=final,
        validation_note="One fixed window/source; provisional split; not a capability benchmark",
        elapsed_seconds=time.monotonic() - start,
        compute_seconds=compute_seconds,
        compute_tokens_per_second=tokens / compute_seconds if compute_seconds else 0,
        checkpoint_seconds=checkpoint_seconds,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        samples=samples,
        final_checkpoint=str(output / f"step-{step:06d}.pt"),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    commit()
    print(json.dumps(report), flush=True)
    return report
