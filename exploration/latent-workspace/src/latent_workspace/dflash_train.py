"""One DFlash-style baseline: target responses, random anchors, and acceptance evaluation."""

import hashlib
import json
import math
import shutil
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .corpus import DOMAINS
from .dflash import DFlashDrafter, block_loss, features, synchronize, verified_generate
from .dflash_data import build_teacher_cache, load_teacher_cache, prepare_prompts
from .train import atomic_save, seed_all, write_json


def precision(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else nullcontext()
    )


@torch.no_grad()
def frozen_memory(target, tokens, lengths, layers):
    mask = torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths[:, None]
    return features(
        target.model(tokens, attention_mask=mask, use_cache=False, output_hidden_states=True),
        layers,
    )


def collate(examples, count, block_size, generator, device, target=None, layers=()):
    length = max(len(e["tokens"]) for e in examples)
    online = target is not None
    memory = (
        None
        if online
        else torch.zeros(
            len(examples),
            length,
            examples[0]["memory"].shape[-1],
            dtype=examples[0]["memory"].dtype,
        )
    )
    tokens = torch.zeros(len(examples), length, dtype=torch.long)
    anchors = []
    for i, e in enumerate(examples):
        n = len(e["tokens"])
        tokens[i, :n] = e["tokens"]
        if not online:
            memory[i, :n] = e["memory"]
        anchors.append(
            torch.randint(e["prompt_length"], n - block_size + 1, (count,), generator=generator)
        )
    tokens = tokens.to(device)
    memory = (
        frozen_memory(
            target,
            tokens,
            torch.tensor([len(e["tokens"]) for e in examples], device=device),
            layers,
        )
        if online
        else memory.to(device)
    )
    return tokens, memory, torch.stack(anchors).to(device)


@torch.no_grad()
def evaluate(drafter, target, examples, mask_id, limit_per_domain=None, include_examples=False):
    drafter.eval()
    result = {}
    example_metrics = []
    for domain in DOMAINS:
        rows = [x for x in examples if x["domain"] == domain]
        if not rows:
            continue
        if limit_per_domain:
            rows = rows[:limit_per_domain]
        ce_sum = torch.zeros(drafter.cfg.block_size - 1)
        hits = torch.zeros_like(ce_sum)
        lengths, blocks = [], 0
        for e in rows:
            first, last = e["prompt_length"], len(e["tokens"]) - drafter.cfg.block_size
            anchors = (
                torch.linspace(first, last, min(8, last - first + 1))
                .round()
                .long()
                .unique()[None]
                .to(target.device)
            )
            with precision(target.device):
                _, ce, predictions, labels = block_loss(
                    drafter,
                    target,
                    e["tokens"][None].to(target.device),
                    e["memory"][None].to(target.device)
                    if "memory" in e
                    else frozen_memory(
                        target,
                        e["tokens"][None].to(target.device),
                        torch.tensor([len(e["tokens"])], device=target.device),
                        drafter.cfg.layers,
                    ),
                    anchors,
                    mask_id,
                )
            ce = ce.view(-1, drafter.cfg.block_size - 1)
            correct = (predictions == labels).view_as(ce)
            ce_sum += ce.sum(0).cpu()
            hits += correct.sum(0).cpu()
            lengths.extend(correct.long().cumprod(-1).sum(-1).cpu().tolist())
            blocks += len(ce)
            if include_examples:
                response = e["tokens"][e["prompt_length"] :].tolist()
                example_metrics.append(
                    {
                        "group": e["group"],
                        "domain": domain,
                        "blocks": len(ce),
                        "response_tokens": len(response),
                        "unique_response_4gram_fraction": len(
                            {tuple(response[i : i + 4]) for i in range(len(response) - 3)}
                        )
                        / max(1, len(response) - 3),
                        "nll": float(ce.mean()),
                        "accepted_beyond_anchor": float(
                            correct.long().cumprod(-1).sum(-1).float().mean()
                        ),
                        "accuracy_by_draft_position": correct.float().mean(0).cpu().tolist(),
                    }
                )
        result[domain] = {
            "nll": float(ce_sum.mean() / blocks),
            "nll_by_draft_position": (ce_sum / blocks).tolist(),
            "accuracy_by_draft_position": (hits / blocks).tolist(),
            "mean_accepted_beyond_anchor": sum(lengths) / len(lengths),
            "full_draft_acceptance": sum(n == drafter.cfg.block_size - 1 for n in lengths)
            / len(lengths),
            "prompts": len(rows),
            "blocks": blocks,
            "acceptance_note": "Consecutive matches on held-out greedy teacher trajectories; anchor excluded. Actual rollout is benchmarked separately.",
        }
    metrics = {
        "by_domain": result,
        "nll": sum(x["nll"] for x in result.values()) / len(result),
        "accepted_beyond_anchor": sum(x["mean_accepted_beyond_anchor"] for x in result.values())
        / len(result),
    }
    if include_examples:
        metrics["examples"] = example_metrics
    return metrics


@torch.no_grad()
def greedy_generate(target, prompt, max_new_tokens, eos_id=None):
    synchronize(prompt.device)
    started = time.perf_counter()
    output = target(prompt, use_cache=True, logits_to_keep=1)
    synchronize(prompt.device)
    prefill = time.perf_counter() - started
    tokens = []
    for i in range(max_new_tokens):
        token = output.logits[:, -1].argmax(-1)
        tokens.append(token.item())
        if token.item() == eos_id or i + 1 == max_new_tokens:
            break
        output = target(token[:, None], past_key_values=output.past_key_values, use_cache=True)
    synchronize(prompt.device)
    return {
        "tokens": tokens,
        "prefill_seconds": prefill,
        "total_seconds": time.perf_counter() - started,
    }


def benchmark(drafter, target, tokenizer, examples, mask_id):
    selected = [
        x
        for d in DOMAINS
        for x in [e for e in examples if e["domain"] == d][: drafter.cfg.rollout_prompts_per_domain]
    ]
    rows = []
    for example in selected:
        prompt = example["tokens"][: example["prompt_length"]][None].to(target.device)
        with precision(target.device) if target.dtype != torch.float32 else nullcontext():
            greedy_generate(target, prompt, 8, tokenizer.eos_token_id)
            verified_generate(drafter, target, prompt, 8, mask_id, tokenizer.eos_token_id)
            ar, spec = [], []
            for repeat in range(3):
                order = ("ar", "spec") if repeat % 2 == 0 else ("spec", "ar")
                for method in order:
                    if method == "ar":
                        ar.append(greedy_generate(target, prompt, 32, tokenizer.eos_token_id))
                    else:
                        spec.append(
                            verified_generate(
                                drafter, target, prompt, 32, mask_id, tokenizer.eos_token_id
                            )
                        )
        match = all(x["tokens"] == ar[0]["tokens"] for x in ar + spec)
        ar_seconds = statistics.median(x["total_seconds"] for x in ar)
        spec_seconds = statistics.median(x["total_seconds"] for x in spec)
        rows.append(
            {
                "domain": example["domain"],
                "group": example["group"],
                "output_tokens": len(ar[0]["tokens"]),
                "identical_to_target_greedy": match,
                "ar_seconds": ar_seconds,
                "speculative_seconds": spec_seconds,
                "end_to_end_speedup": ar_seconds / spec_seconds if match else None,
                "target_arithmetic": str(target.dtype),
                "mismatch_tokens": None
                if match
                else {"ar": ar[0]["tokens"], "speculative": spec[0]["tokens"]},
                "accepted_beyond_anchor": spec[0]["mean_accepted_beyond_anchor"],
                "acceptance_lengths": spec[0]["accepted_new_draft_tokens"],
                "verification_calls": spec[0]["verification_calls"],
                "text": tokenizer.decode(ar[0]["tokens"]),
                "timing_note": "Median of 3 alternating-order warmed runs, includes prefill; batch 1, 32-token cap, same target/precision. Small validation slice, not a general speed benchmark.",
            }
        )
    return rows


def restore_continuation(cfg, previous, state, drafter, optimizer, generator):
    old = previous["config"]
    old_domains = tuple(old.get("training_domains", DOMAINS))
    if cfg.curriculum_switch:
        if (
            old_domains != ("prose", "code")
            or tuple(cfg.training_domains) != tuple(DOMAINS)
            or old["train_limit"] != 131072
            or cfg.train_limit != 196608
            or cfg.teacher_parent == old["teacher_parent"]
        ):
            raise ValueError("invalid declared prose/code to full-data curriculum switch")
    elif tuple(cfg.training_domains) != old_domains:
        raise ValueError("continuation changes training domains without curriculum switch")
    for key, default in (
        ("teacher_parent", ""),
        ("prose_prefixes_per_article", 1),
        ("online_teacher_features", False),
        ("training_response_cap", 0),
        ("train_limit", 0),
        ("schedule_steps", 0),
        ("teacher_prompt_style", "raw"),
        ("qk_norm", False),
        ("warm_start_from", ""),
        ("retain_training_parent", ""),
    ):
        if cfg.curriculum_switch and key in ("teacher_parent", "train_limit"):
            continue
        if getattr(cfg, key) != old.get(key, default):
            raise ValueError(f"continuation changes cached subset: {key}")
    for key in (
        "model",
        "model_revision",
        "layers",
        "draft_layers",
        "block_size",
        "gamma",
        "train_prompts",
        "validation_prompts",
        "test_prompts",
        "prompt_tokens",
        "response_tokens",
        "batch_size",
        "anchors_per_sequence",
        "learning_rate",
        "seed",
    ):
        if json.dumps(cfg.to_dict()[key]) != json.dumps(old[key]):
            raise ValueError(f"continuation changes baseline/data: {key}")
    if state["step"] != previous["curve"][-1]["step"]:
        raise ValueError("checkpoint/report step mismatch")
    drafter.load_state_dict(state["drafter"])
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["sampler_rng"].cpu())
    torch.set_rng_state(state["rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])


def run_dflash(cfg, output, on_checkpoint=None):
    cfg.validate()
    started = time.monotonic()
    deadline = started + cfg.max_seconds
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_all(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model,
            revision=cfg.model_revision,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    mask_id = tokenizer.get_vocab().get("<|fim_pad|>", tokenizer.eos_token_id)
    previous, state = None, None
    if cfg.continue_from and not cfg.curriculum_switch:
        parent = Path(cfg.continue_from)
        previous = json.loads((parent / "report.json").read_text())
        state = torch.load(parent / "latest.pt", map_location="cpu", weights_only=True)
        data = load_teacher_cache(parent / "teacher.pt")
        audit = json.loads((parent / "teacher-audit.json").read_text())
        for name in ("teacher.pt", "teacher-audit.json", "prompts.json", "best.pt"):
            shutil.copyfile(parent / name, output / name)
    elif cfg.teacher_parent:
        from .dflash_scaling import nested_subset, teacher_statistics

        parent = Path(cfg.teacher_parent)
        if cfg.curriculum_switch:
            previous_parent = Path(cfg.continue_from)
            previous = json.loads((previous_parent / "report.json").read_text())
            state = torch.load(previous_parent / "latest.pt", map_location="cpu", weights_only=True)
            shutil.copyfile(previous_parent / "best.pt", output / "best.pt")
        if not json.loads((parent / "report.json").read_text()).get("completed"):
            raise ValueError("teacher preparation is incomplete")
        parent_cfg = json.loads((parent / "config.json").read_text())
        for key in (
            "model",
            "model_revision",
            "layers",
            "prompt_tokens",
            "response_tokens",
            "block_size",
            "teacher_prompt_style",
        ):
            if json.dumps(cfg.to_dict()[key]) != json.dumps(
                parent_cfg.get(key, "raw") if key == "teacher_prompt_style" else parent_cfg[key]
            ):
                raise ValueError(f"teacher cache/config mismatch: {key}")
        data = load_teacher_cache(parent / "teacher.pt")
        required = set()
        retain_parents = [cfg.retain_training_parent] if cfg.retain_training_parent else []
        if cfg.curriculum_switch:
            retain_parents.append(cfg.continue_from)
        for retain_parent in retain_parents:
            retained = load_teacher_cache(Path(retain_parent) / "teacher.pt")
            available = {e["group"]: e for e in data["train"]}
            for e in retained["train"]:
                if e["domain"] not in cfg.training_domains:
                    continue
                if e["group"] not in available or not torch.equal(
                    e["tokens"],
                    available[e["group"]]["tokens"][: len(e["tokens"])]
                    if cfg.online_teacher_features
                    else available[e["group"]]["tokens"],
                ):
                    raise ValueError("required training example missing or labels changed")
                required.add(e["group"])
            del retained, available
        data["train"] = nested_subset(
            data["train"], cfg.train_limit, cfg.seed, required, domains=cfg.training_domains
        )
        if cfg.training_response_cap:
            data["train"] = [
                {
                    **e,
                    "tokens": e["tokens"][: e["prompt_length"] + cfg.training_response_cap],
                    **(
                        {"memory": e["memory"][: e["prompt_length"] + cfg.training_response_cap]}
                        if "memory" in e
                        else {}
                    ),
                }
                for e in data["train"]
            ]
        audit = teacher_statistics(data, cfg.block_size)
        audit["required_training_groups"] = len(required)
        audit["parent_generation_audit"] = json.loads((parent / "teacher-audit.json").read_text())
        atomic_save(output / "teacher.pt", data)
        write_json(output / "teacher-audit.json", audit)
        shutil.copyfile(parent / "prompts.json", output / "prompts.json")
    else:
        manifest = prepare_prompts(cfg, tokenizer, output, deadline - 120)
        if on_checkpoint:
            on_checkpoint()
        data, audit = build_teacher_cache(
            cfg, target, tokenizer, manifest, output, deadline - 120, on_checkpoint
        )
    seed_all(cfg.seed)
    drafter = DFlashDrafter(target.config, cfg).to(device)
    warm_start = None
    if cfg.warm_start_from and not previous:
        path = Path(cfg.warm_start_from) / "best.pt"
        warm_state = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("model", "model_revision", "layers", "draft_layers", "qk_norm", "block_size"):
            default = False if key == "qk_norm" else None
            if json.dumps(cfg.to_dict()[key]) != json.dumps(warm_state["config"].get(key, default)):
                raise ValueError(f"warm start changes draft architecture: {key}")
        if mask_id != warm_state["mask_token_id"]:
            raise ValueError("warm start changes mask token")
        drafter.load_state_dict(warm_state["drafter"])
        with path.open("rb") as f:
            checkpoint_sha = hashlib.file_digest(f, "sha256").hexdigest()
        warm_start = {
            "parent": cfg.warm_start_from,
            "step": warm_state["step"],
            "checkpoint_sha256": checkpoint_sha,
            "note": "Weights only; fresh AdamW and new schedule on the declared teacher corpus.",
        }
        del warm_state
    optimizer = torch.optim.AdamW(drafter.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(cfg.seed)
    start_step = 0
    if previous:
        restore_continuation(cfg, previous, state, drafter, optimizer, generator)
        if mask_id != previous["mask_token_id"]:
            raise ValueError("continuation changes mask token")
        start_step = state["step"]
    if cfg.schedule_steps and start_step + cfg.steps > cfg.schedule_steps:
        raise ValueError("requested updates exceed the fixed global schedule")
    source = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        source.update(file.name.encode() + file.read_bytes())
    report = {
        "config": cfg.to_dict(),
        "source_sha256": source.hexdigest(),
        "mask_token_id": mask_id,
        "trainable_parameters": sum(p.numel() for p in drafter.parameters()),
        "teacher_audit": audit,
        "curve": [],
        "completed": False,
        "test_status": "new test prompts reserved; no teacher test responses/features/metrics produced",
        "data_note": "All labels are pinned frozen Qwen greedy continuations, not original human answers.",
        "warm_start": previous.get("warm_start") if previous else warm_start,
    }
    initial_weights = hashlib.sha256()
    for name, tensor in sorted(drafter.state_dict().items()):
        initial_weights.update(name.encode() + tensor.detach().cpu().numpy().tobytes())
    report["worker_start_drafter_sha256"] = initial_weights.hexdigest()
    report["initial_drafter_sha256"] = (
        previous["initial_drafter_sha256"] if previous else initial_weights.hexdigest()
    )
    pools = {d: [e for e in data["train"] if e["domain"] == d] for d in cfg.training_domains}
    if any(not pool for pool in pools.values()):
        raise ValueError("training domain has no usable examples")
    best = (-1.0, -float("inf"))
    supervised = 0
    sampled_anchors = 0
    if previous:
        report["curve"] = previous["curve"]
        report["parent_run"] = cfg.continue_from
        report["parent_source_sha256"] = previous["source_sha256"]
        report["best_step"] = previous["best_step"]
        selected = next(p for p in previous["curve"] if p["step"] == previous["best_step"])[
            "validation"
        ]
        best = (selected["accepted_beyond_anchor"], -selected["nll"])
        supervised = previous["curve"][-1]["supervised_target_exposures"]
        sampled_anchors = previous["curve"][-1]["sampled_anchor_blocks"]
    report["training_goal_step"] = start_step + cfg.steps
    report["schedule_note"] = (
        "Each run is a new warmup/cosine cycle; continuation restores model, AdamW and RNG on the exact archived teacher corpus."
    )
    sequence_hash = hashlib.sha256()
    report["split_sequence_sha256"] = {}
    for split, examples in data.items():
        split_hash = hashlib.sha256()
        for e in examples:
            identity = split.encode() + e["group"].encode() + e["tokens"].numpy().tobytes()
            sequence_hash.update(identity)
            split_hash.update(identity)
        report["split_sequence_sha256"][split] = split_hash.hexdigest()
    report["teacher_sequence_sha256"] = sequence_hash.hexdigest()
    if cfg.curriculum_switch:
        for split in ("validation", "validation_short"):
            if report["split_sequence_sha256"].get(split) != previous["split_sequence_sha256"].get(
                split
            ):
                raise ValueError("curriculum switch changes held-out data")
        report["curriculum_transition"] = {
            "step": start_step,
            "from_domains": previous["config"]["training_domains"],
            "to_domains": list(cfg.training_domains),
            "from_examples": previous["config"]["train_limit"],
            "to_examples": cfg.train_limit,
            "optimizer_and_rng_restored": True,
        }
    elif previous and previous.get("curriculum_transition"):
        report["curriculum_transition"] = previous["curriculum_transition"]
    for cycle_step in range(1 if previous else 0, cfg.steps + 1):
        step = start_step + cycle_step
        gradient_diagnostic = None
        if cycle_step:
            drafter.train()
            examples = []
            for i in range(cfg.batch_size):
                pool = pools[cfg.training_domains[((step - 1) * cfg.batch_size + i) % len(pools)]]
                examples.append(pool[int(torch.randint(len(pool), (), generator=generator))])
            tokens, memory, anchors = collate(
                examples,
                cfg.anchors_per_sequence,
                cfg.block_size,
                generator,
                device,
                target=target if cfg.online_teacher_features else None,
                layers=cfg.layers,
            )
            schedule_steps = cfg.schedule_steps or cfg.steps
            schedule_step = step if cfg.schedule_steps else cycle_step
            warmup = max(1, round(schedule_steps * cfg.warmup_ratio))
            scale = (
                min(schedule_step / warmup, 1.0)
                if schedule_step <= warmup
                else 0.5
                * (
                    1
                    + math.cos(
                        math.pi
                        * min(1.0, (schedule_step - warmup) / max(1, schedule_steps - warmup))
                    )
                )
            )
            for group in optimizer.param_groups:
                group["lr"] = cfg.learning_rate * scale
            optimizer.zero_grad(set_to_none=True)
            with precision(device):
                loss, _, _, _ = block_loss(drafter, target, tokens, memory, anchors, mask_id)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite DFlash-style training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
            if cycle_step % cfg.eval_every == 0 or cycle_step == cfg.steps:
                gradient_diagnostic = {
                    "global_norm_before_clip": float(grad_norm),
                    "feature_projection_norm_after_clip": float(drafter.fc.weight.grad.norm()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "batch_loss": float(loss.detach()),
                }
            optimizer.step()
            sampled_anchors += anchors.numel()
            supervised += anchors.numel() * (cfg.block_size - 1)
        time_limit = time.monotonic() >= deadline - 60
        if cycle_step % cfg.eval_every == 0 or cycle_step == cfg.steps or time_limit:
            validation = evaluate(drafter, target, data["validation"], mask_id)
            training = evaluate(drafter, target, data["train"], mask_id, limit_per_domain=4)
            row = {
                "step": step,
                "validation": validation,
                "training_diagnostic": training,
                "sampled_anchor_blocks": sampled_anchors,
                "supervised_target_exposures": supervised,
                "elapsed_seconds": time.monotonic() - started,
                "phase": output.name,
                "training_domains": list(cfg.training_domains),
                "training_examples": len(data["train"]),
                "gradient_diagnostic": gradient_diagnostic,
            }
            if "validation_short" in data:
                row["validation_short"] = evaluate(
                    drafter, target, data["validation_short"], mask_id
                )
            report["curve"].append(row)
            state = {
                "drafter": drafter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": cfg.to_dict(),
                "mask_token_id": mask_id,
                "rng": torch.get_rng_state(),
                "sampler_rng": generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                "source_sha256": report["source_sha256"],
            }
            atomic_save(output / "latest.pt", state)
            score = (validation["accepted_beyond_anchor"], -validation["nll"])
            if score > best:
                best = score
                report["best_step"] = step
                atomic_save(
                    output / "best.pt", {k: v for k, v in state.items() if k != "optimizer"}
                )
            write_json(output / "report.json", report)
            if on_checkpoint:
                on_checkpoint()
            print(
                f"DFlash-style step={step} accepted NEW={validation['accepted_beyond_anchor']:.3f}/{cfg.block_size - 1} nll={validation['nll']:.4f}",
                flush=True,
            )
        if time_limit:
            report["stopped"] = "bounded time limit; training incomplete"
            break
    state = torch.load(output / "best.pt", map_location=device, weights_only=True)
    drafter.load_state_dict(state["drafter"])
    report["selected_validation"] = evaluate(
        drafter, target, data["validation"], mask_id, include_examples=True
    )
    report["training_completed"] = cycle_step == cfg.steps
    report["training_peak_cuda_memory_bytes"] = (
        torch.cuda.max_memory_allocated() if device == "cuda" else None
    )
    if cfg.quality_only:
        report["rollout_benchmark"] = []
        report["verification_passed"] = None
        report["verification_status"] = (
            "not run: quality-only experiment; hybrid cache rollback unsupported"
        )
        report["completed"] = report["training_completed"]
    else:
        # Correctness reference: FP32 on the same stored BF16 weights.
        target.float()
        report["verification_target_arithmetic"] = "float32; stored BF16 weights upcast"
        report["rollout_benchmark"] = benchmark(
            drafter, target, tokenizer, data["validation"], mask_id
        )
        report["verification_passed"] = all(
            x["identical_to_target_greedy"] for x in report["rollout_benchmark"]
        )
        report["completed"] = report["training_completed"] and report["verification_passed"]
    report["elapsed_seconds"] = time.monotonic() - started
    report["frozen_target_has_no_gradients"] = all(p.grad is None for p in target.parameters())
    write_json(output / "report.json", report)
    return report
