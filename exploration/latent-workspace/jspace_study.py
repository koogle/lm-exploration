"""Matched J-space augmentation study; frozen package and historical test untouched."""

import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from jspace_next import j_dictionary, sparse_component
from latent_workspace.corpus import DOMAINS
from latent_workspace.dflash import DFlashDrafter, DraftConfig, block_loss
from latent_workspace.dflash_data import load_teacher_cache, token_hash
from latent_workspace.dflash_scaling import diverse_manifest, generate_batch
from latent_workspace.dflash_train import collate, evaluate, frozen_memory, precision
from latent_workspace.train import atomic_save, seed_all, write_json

ARMS = ("raw", "jspace", "random")


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class AugmentedDrafter(DFlashDrafter):
    """Frozen capable raw projection plus an equally sized zero-initialized correction."""

    def __init__(self, target_config, cfg, state):
        super().__init__(target_config, cfg)
        self.load_state_dict(state)
        self.fc.requires_grad_(False)
        self.correction = nn.Linear(self.fc.in_features, self.fc.out_features, bias=False)
        nn.init.zeros_(self.correction.weight)

    def context_cache(self, memory, positions, previous=None):
        width = self.fc.in_features
        if memory.shape[-1] != 2 * width:
            raise ValueError("Expected unchanged raw states plus auxiliary features")
        raw, side = memory.split(width, dim=-1)
        x = self.hidden_norm(self.fc(raw) + self.correction(side))
        cos, sin = self.rotary(x, positions)
        cache = []
        for i, layer in enumerate(self.layers):
            kv = layer.kv(x, cos, sin)
            if previous is not None:
                kv = tuple(torch.cat((old, new), dim=-2) for old, new in zip(previous[i], kv))
            cache.append(kv)
        return cache


class Projector:
    def __init__(self, target, calibration, protocol):
        saved = torch.load(Path(calibration) / "jacobians.pt", weights_only=True)
        if saved["prompts"] != protocol["calibration_prompts"]:
            raise ValueError("Incomplete calibration must not enter the quality comparison")
        if saved["layers"] != protocol["layers"]:
            raise ValueError("Calibration layer mismatch")
        self.protocol = protocol
        self.width = target.config.hidden_size
        self.j, self.random = [], []
        generator = torch.Generator(device="cuda").manual_seed(protocol["random_frame_seed"])
        for jacobian in saved["matrices"]:
            dictionary = j_dictionary(
                target.get_output_embeddings().weight, target.model.norm.weight, jacobian.cuda()
            )
            rotation, _ = torch.linalg.qr(
                torch.randn(self.width, self.width, generator=generator, device="cuda")
            )
            self.j.append(dictionary.to(torch.bfloat16))
            self.random.append((dictionary @ rotation).to(torch.bfloat16))
        self.selection_sha = saved["selection_sha256"]

    @torch.no_grad()
    def project(self, memory, kind):
        flat = memory.reshape(-1, memory.shape[-1])
        result = torch.empty_like(flat)
        dictionaries = self.j if kind == "jspace" else self.random
        chunk = self.protocol["projection_chunk"]
        for depth, dictionary in enumerate(dictionaries):
            span = slice(depth * self.width, (depth + 1) * self.width)
            for start in range(0, len(flat), chunk):
                end = min(start + chunk, len(flat))
                result[start:end, span] = sparse_component(
                    flat[start:end, span],
                    dictionary,
                    k=self.protocol["sparse_k"],
                    shortlist=self.protocol["shortlist"],
                    refit_steps=self.protocol["refit_steps"],
                )[0]
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite auxiliary conditioning")
        return result.reshape_as(memory)


def prepare_holdout(protocol, output, commit):
    """Sample fresh source-disjoint validation/test prompts; leave test labels unopened."""
    from transformers import AutoTokenizer

    output = Path(output)
    output.mkdir(exist_ok=False)
    historical = load_teacher_cache("/runs/data-large-01/corpus.pt")["metadata"]
    prior = json.loads((Path(protocol["manifest_parent"]) / "prompts.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model"], revision=protocol["model_revision"]
    )
    # Three registered training placeholders satisfy the sampler's split contract;
    # they are never used as new training examples or held-out data.
    placeholders = [
        next(r for r in prior["records"] if r["domain"] == d and r["split"] == "train")
        for d in DOMAINS
    ]
    cfg = DraftConfig(
        model=protocol["model"],
        model_revision=protocol["model_revision"],
        train_prompts=3,
        validation_prompts=protocol["validation_prompts"],
        test_prompts=protocol["test_prompts"],
        prose_prefixes_per_article=4,
        seed=protocol["holdout_seed"],
        prompt_tokens=128,
    )
    manifest = diverse_manifest(
        cfg,
        tokenizer,
        output,
        time.monotonic() + 3300,
        historical,
        prior,
        initial_records=placeholders,
        on_partial=lambda rows: write_json(output / "partial.json", {"records": rows}),
    )
    prior_groups = {r.get("source_group", r["group"]) for r in prior["records"]}
    new = [r for r in manifest["records"] if r["split"] != "train"]
    if any(r.get("source_group", r["group"]) in prior_groups for r in new):
        raise ValueError("Fresh held-out sources overlap a prior split")
    write_json(
        output / "report.json",
        {
            "completed": True,
            "new_validation": protocol["validation_prompts"],
            "new_test": protocol["test_prompts"],
            "prior_manifest_sha256": file_sha(Path(protocol["manifest_parent"]) / "prompts.json"),
            "training_placeholders_not_used": True,
            "test_labels_status": "unopened",
        },
    )
    commit()
    return str(output)


def validation_features(target, tokenizer, projector, protocol, output, commit, split="validation"):
    """Generate only fresh validation responses, deduplicated against old data."""
    manifest = json.loads((Path(protocol["holdout_parent"]) / "prompts.json").read_text())
    if split == "test" and not (output / "frozen.json").exists():
        raise ValueError("Freeze selected checkpoints before opening fresh test labels")
    records = [r for r in manifest["records"] if r["split"] == split]
    teacher = load_teacher_cache(Path(protocol["teacher_parent"]) / "teacher.pt")
    existing = {
        token_hash(e["tokens"][e["prompt_length"] :].tolist())
        for split in ("train", "validation")
        for e in teacher[split]
    }
    if split == "test":
        fresh_validation = load_teacher_cache(output / "validation-features.pt")["examples"]
        existing.update(
            token_hash(e["tokens"][e["prompt_length"] :].tolist()) for e in fresh_validation
        )
    examples, excluded = [], []
    for start in range(0, len(records), protocol["teacher_batch_size"]):
        batch = records[start : start + protocol["teacher_batch_size"]]
        sequences = generate_batch(
            target, batch, protocol["response_tokens"], tokenizer.eos_token_id
        )
        for record, sequence in zip(batch, sequences):
            response = sequence[len(record["prompt_ids"]) :]
            key = token_hash(response)
            if len(response) < 8 or key in existing:
                excluded.append(
                    {
                        "group": record["group"],
                        "domain": record["domain"],
                        "reason": "short" if len(response) < 8 else "response_duplicate",
                    }
                )
                continue
            existing.add(key)
            tokens = torch.tensor(sequence)
            raw = frozen_memory(
                target,
                tokens[None].cuda(),
                torch.tensor([len(tokens)], device="cuda"),
                protocol["layers"],
            )
            row = {
                "tokens": tokens,
                "prompt_length": len(record["prompt_ids"]),
                "group": record["group"],
                "domain": record["domain"],
            }
            row["raw"] = raw[0].cpu()
            row["jspace"] = projector.project(raw, "jspace")[0].cpu()
            row["random"] = projector.project(raw, "random")[0].cpu()
            examples.append(row)
        print(
            f"Fresh {split} features {min(start + len(batch), len(records))}/{len(records)}",
            flush=True,
        )
    if any(sum(e["domain"] == d for e in examples) < 32 for d in DOMAINS):
        raise ValueError(f"Insufficient fresh {split} cases after fixed exclusions")
    atomic_save(output / f"{split}-features.pt", {"examples": examples, "excluded": excluded})
    commit()
    return examples, excluded


def side_examples(examples, arm):
    return [
        {**e, "memory": torch.cat((e["raw"], e["raw"] if arm == "raw" else e[arm]), -1)}
        for e in examples
    ]


def compare_chunk(protocol, calibration, output, commit, resume=""):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(output)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    parent = Path(protocol["checkpoint_parent"])
    if file_sha(parent / "best.pt") != protocol["checkpoint_sha256"]:
        raise ValueError("Qualified checkpoint changed")
    original = torch.load(parent / "best.pt", map_location="cpu", weights_only=True)
    cfg = DraftConfig(**original["config"]).validate()
    if original["step"] != protocol["checkpoint_step"] or list(cfg.layers) != protocol["layers"]:
        raise ValueError("Wrong capable initialization")
    seed_all(protocol["seed"])
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model, revision=cfg.model_revision, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    drafts = {arm: AugmentedDrafter(target.config, cfg, original["drafter"]).cuda() for arm in ARMS}
    optimizers = {
        arm: torch.optim.AdamW(
            [p for p in draft.parameters() if p.requires_grad],
            lr=protocol["learning_rate"],
            weight_decay=0.01,
        )
        for arm, draft in drafts.items()
    }
    generator = torch.Generator().manual_seed(protocol["seed"])
    report = {
        "protocol": protocol,
        "calibration_parent": calibration,
        "curve": [],
        "completed": False,
        "source_sha256": file_sha(__file__),
        "best_steps": {arm: 0 for arm in ARMS},
        "test_status": "Fresh test labels unopened; historical drafting test not evaluated",
        "trainable_parameters": {
            arm: sum(p.numel() for p in draft.parameters() if p.requires_grad)
            for arm, draft in drafts.items()
        },
    }
    best = {arm: (-1.0, -math.inf) for arm in ARMS}
    start_step = 0
    if resume:
        state = torch.load(Path(resume) / "latest.pt", weights_only=True, map_location="cpu")
        if state["protocol"] != protocol or state["source_sha256"] != report["source_sha256"]:
            raise ValueError("Resume changes training protocol/source")
        for arm in ARMS:
            drafts[arm].load_state_dict(state["drafts"][arm])
            optimizers[arm].load_state_dict(state["optimizers"][arm])
            (output / f"best-{arm}.pt").write_bytes((Path(resume) / f"best-{arm}.pt").read_bytes())
        generator.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step, best = state["step"], state["best"]
        report = json.loads((Path(resume) / "report.json").read_text())
        report["resumed_from"] = resume
        del state
    del original
    projector = Projector(target, calibration, protocol)
    report["calibration_selection_sha256"] = projector.selection_sha
    data = load_teacher_cache(parent / "teacher.pt")
    training = data["train"]
    if len(training) != protocol["training_examples"]:
        raise ValueError("Changed training pool")
    pools = {d: [e for e in training if e["domain"] == d] for d in DOMAINS}
    cached = (
        Path(resume) / "validation-features.pt" if resume else output / "validation-features.pt"
    )
    if cached.exists():
        (output / "validation-features.pt").write_bytes(
            cached.read_bytes()
        ) if cached.parent != output else None
        cache = load_teacher_cache(output / "validation-features.pt")
        validation, excluded = cache["examples"], cache["excluded"]
    else:
        validation, excluded = validation_features(
            target, tokenizer, projector, protocol, output, commit
        )
    report["validation_cases"] = {d: sum(e["domain"] == d for e in validation) for d in DOMAINS}
    report["validation_exclusions"] = excluded
    report["labels_per_update_per_arm"] = (
        protocol["batch_size"] * protocol["anchors_per_sequence"] * 7
    )
    eval_rows = {arm: side_examples(validation, arm) for arm in ARMS}
    sample_digest = hashlib.sha256()
    projection_seconds = 0.0

    def checkpoint(step, gradients):
        scores = {
            arm: evaluate(drafts[arm], target, eval_rows[arm], original_mask, include_examples=True)
            for arm in ARMS
        }
        if step == 0:
            vals = [scores[arm]["accepted_beyond_anchor"] for arm in ARMS]
            if max(vals) - min(vals) > 1e-6:
                raise ValueError("Zero-initialized arms do not start identically")
        for arm in ARMS:
            score = (scores[arm]["accepted_beyond_anchor"], -scores[arm]["nll"])
            if score > tuple(best[arm]):
                best[arm] = score
                report["best_steps"][arm] = step
                atomic_save(
                    output / f"best-{arm}.pt",
                    {"drafter": drafts[arm].state_dict(), "step": step, "protocol": protocol},
                )
        report["curve"].append(
            {
                "step": step,
                "validation": scores,
                "gradients": gradients,
                "new_label_exposures_per_arm": step * report["labels_per_update_per_arm"],
                "elapsed_chunk_seconds": time.monotonic() - started,
                "projection_seconds": projection_seconds,
                "sample_stream_chunk_sha256": sample_digest.hexdigest(),
            }
        )
        report["step"] = step
        report["completed"] = step == protocol["steps"]
        report["target_gradients_absent"] = all(p.grad is None for p in target.parameters())
        atomic_save(
            output / "latest.pt",
            {
                "drafts": {a: d.state_dict() for a, d in drafts.items()},
                "optimizers": {a: o.state_dict() for a, o in optimizers.items()},
                "step": step,
                "best": best,
                "sampler_rng": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "protocol": protocol,
                "source_sha256": file_sha(__file__),
            },
        )
        write_json(output / "report.json", report)
        commit()
        print(
            "Matched study",
            step,
            {a: round(s["accepted_beyond_anchor"], 4) for a, s in scores.items()},
            flush=True,
        )

    original_mask = json.loads((parent / "report.json").read_text())["mask_token_id"]
    if not resume:
        checkpoint(0, {})
    gradients = {}
    for step in range(start_step + 1, protocol["steps"] + 1):
        if time.monotonic() - started > protocol["max_seconds"] - 240:
            if report.get("step", -1) != step - 1:
                checkpoint(step - 1, gradients)
            break
        examples = []
        for i in range(protocol["batch_size"]):
            pool = pools[DOMAINS[((step - 1) * protocol["batch_size"] + i) % 3]]
            examples.append(pool[int(torch.randint(len(pool), (), generator=generator))])
        tokens, raw, anchors = collate(
            examples,
            protocol["anchors_per_sequence"],
            8,
            generator,
            "cuda",
            target=target,
            layers=cfg.layers,
        )
        sample_digest.update(
            "".join(e["group"] for e in examples).encode() + anchors.cpu().numpy().tobytes()
        )
        torch.cuda.synchronize()
        t = time.monotonic()
        sides = {
            "raw": raw,
            "jspace": projector.project(raw, "jspace"),
            "random": projector.project(raw, "random"),
        }
        torch.cuda.synchronize()
        projection_seconds += time.monotonic() - t
        warmup = protocol["warmup_steps"]
        scale = (
            min(step / warmup, 1.0)
            if step <= warmup
            else 0.5 * (1 + math.cos(math.pi * (step - warmup) / (protocol["steps"] - warmup)))
        )
        gradients = {}
        for arm in ARMS:
            draft, optimizer = drafts[arm], optimizers[arm]
            draft.train()
            for group in optimizer.param_groups:
                group["lr"] = protocol["learning_rate"] * scale
            optimizer.zero_grad(set_to_none=True)
            with precision("cuda"):
                loss, _, _, _ = block_loss(
                    draft, target, tokens, torch.cat((raw, sides[arm]), -1), anchors, original_mask
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {arm}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            gradients[arm] = {
                "loss": float(loss.detach()),
                "global_norm_before_clip": float(norm),
                "correction_grad_norm": float(draft.correction.weight.grad.norm()),
                "correction_weight_norm": float(draft.correction.weight.norm()),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if not math.isfinite(float(norm)):
                raise FloatingPointError(f"Nonfinite gradients in {arm}")
            optimizer.step()
        if step % protocol["eval_every"] == 0 or step == protocol["steps"]:
            checkpoint(step, gradients)
    return {
        "output": str(output),
        "completed": report["completed"],
        "step": report.get("step", 0),
        "best_steps": report["best_steps"],
    }


def profile(protocol, output, commit):
    """Real frozen checkpoint, causal leakage check and representative step timing."""
    from transformers import AutoModelForCausalLM

    output = Path(output)
    output.mkdir(exist_ok=False)
    state = torch.load(Path(protocol["checkpoint_parent"]) / "best.pt", weights_only=True)
    cfg = DraftConfig(**state["config"])
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model, revision=cfg.model_revision, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    drafts = {a: AugmentedDrafter(target.config, cfg, state["drafter"]).cuda() for a in ARMS}
    teacher = load_teacher_cache(Path(protocol["checkpoint_parent"]) / "teacher.pt")
    rows = [
        next(e for e in teacher["train"] if e["domain"] == DOMAINS[i % 3])
        for i in range(protocol["batch_size"])
    ]
    generator = torch.Generator().manual_seed(42)
    tokens, raw, anchors = collate(
        rows,
        protocol["anchors_per_sequence"],
        8,
        generator,
        "cuda",
        target=target,
        layers=cfg.layers,
    )
    # Two-prefix matrices are permitted only for this plumbing/timing check.
    projector = Projector(
        target, "/runs/jspace-capable-check-02", {**protocol, "calibration_prompts": 2}
    )
    torch.cuda.synchronize()
    t = time.monotonic()
    side = {
        "raw": raw,
        "jspace": projector.project(raw, "jspace"),
        "random": projector.project(raw, "random"),
    }
    torch.cuda.synchronize()
    projection_time = time.monotonic() - t
    outputs, times, gradients = {}, {}, {}
    mask_id = state["mask_token_id"]
    for arm in ARMS:
        t = time.monotonic()
        optimizer = torch.optim.AdamW(
            [p for p in drafts[arm].parameters() if p.requires_grad], lr=protocol["learning_rate"]
        )
        with precision("cuda"):
            out = block_loss(
                drafts[arm], target, tokens, torch.cat((raw, side[arm]), -1), anchors, mask_id
            )
        outputs[arm] = out[2].detach()
        out[0].backward()
        gradients[arm] = float(drafts[arm].correction.weight.grad.norm())
        optimizer.step()
        torch.cuda.synchronize()
        times[arm] = time.monotonic() - t
    equal = all(torch.equal(outputs["raw"], outputs[a]) for a in ARMS)
    if not equal or any(not math.isfinite(x) or x <= 0 for x in gradients.values()):
        raise ValueError("Initial parity or auxiliary gradient flow failed")
    # Perturb only the first anchor's future features, including its own anchor.
    anchor = int(anchors[0, 0])
    draft = drafts["jspace"].eval()
    mem = torch.cat((raw[:1], side["jspace"][:1]), -1)
    with torch.no_grad(), precision("cuda"):
        first = block_loss(draft, target, tokens[:1], mem, anchors[:1, :1], mask_id)
        altered = mem.clone()
        altered[:, anchor:] = torch.randn_like(altered[:, anchor:]) * 20
        second = block_loss(draft, target, tokens[:1], altered, anchors[:1, :1], mask_id)
    leakage = float((first[1] - second[1]).abs().max())
    if leakage != 0:
        raise ValueError("Auxiliary future context leaks into predictions")
    report = {
        "initial_predictions_identical": equal,
        "correction_gradient_norms": gradients,
        "future_feature_perturbation_max_nll_difference": leakage,
        "projection_seconds": projection_time,
        "train_step_seconds_by_arm": times,
        "estimated_30k_hours_excluding_loading_eval_checkpoint": (
            projection_time + sum(times.values())
        )
        * 30000
        / 3600,
        "batch_shape": list(raw.shape),
        "anchors_per_sequence": protocol["anchors_per_sequence"],
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "target_gradients_absent": all(p.grad is None for p in target.parameters()),
        "calibration_prompts": 2,
        "note": "Plumbing/timing check only; not a conditioning benefit measurement",
    }
    write_json(output / "report.json", report)
    commit()
    return report


def paired_intervals(metrics, iterations=5000, seed=20261013):
    """Paired source-group bootstrap of equal-domain, block-weighted means."""
    import numpy as np

    rng = np.random.default_rng(seed)
    keys = ("jspace-minus-raw", "random-minus-raw", "jspace-minus-random")
    pairs = (("jspace", "raw"), ("random", "raw"), ("jspace", "random"))
    samples = {key: np.zeros(iterations) for key in keys}
    for domain in DOMAINS:
        rows = {
            arm: [e for e in m["examples"] if e["domain"] == domain]
            for arm, m in metrics.items()
            if arm in ARMS
        }
        ids = [r["group"] for r in rows["raw"]]
        if any([r["group"] for r in rows[arm]] != ids for arm in ARMS):
            raise ValueError("Paired evaluation source identities differ")
        blocks = np.array([r["blocks"] for r in rows["raw"]])
        indices = rng.integers(0, len(ids), size=(iterations, len(ids)))
        denominator = blocks[indices].sum(-1)
        scores = {
            arm: (np.array([r["accepted_beyond_anchor"] for r in rows[arm]]) * blocks)[indices].sum(
                -1
            )
            / denominator
            for arm in ARMS
        }
        for key, (first, second) in zip(keys, pairs):
            samples[key] += (scores[first] - scores[second]) / 3
    return {
        key: {
            "delta": metrics[first]["accepted_beyond_anchor"]
            - metrics[second]["accepted_beyond_anchor"],
            "ci95": np.quantile(samples[key], [0.025, 0.975]).tolist(),
        }
        for key, (first, second) in zip(keys, pairs)
    }


def finish_test(protocol, calibration, parent, output, commit):
    """Freeze all validation-selected arms, then evaluate a fresh paired test once."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(output)
    output.mkdir(exist_ok=False)
    parent = Path(parent)
    training = json.loads((parent / "report.json").read_text())
    if not training["completed"] or training["protocol"] != protocol:
        raise ValueError("Matched training incomplete or protocol changed")
    frozen = {
        "protocol": protocol,
        "checkpoint_sha256": {arm: file_sha(parent / f"best-{arm}.pt") for arm in ARMS},
        "best_steps": training["best_steps"],
        "training_report_sha256": file_sha(parent / "report.json"),
        "test_status": "UNOPENED",
        "no_test_based_tuning": True,
    }
    write_json(output / "frozen.json", frozen)
    commit()
    cfg_state = torch.load(Path(protocol["checkpoint_parent"]) / "best.pt", weights_only=True)
    cfg = DraftConfig(**cfg_state["config"])
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model, revision=cfg.model_revision, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    projector = Projector(target, calibration, protocol)
    (output / "validation-features.pt").write_bytes(
        (parent / "validation-features.pt").read_bytes()
    )
    examples, excluded = validation_features(
        target, tokenizer, projector, protocol, output, commit, split="test"
    )
    metrics = {}
    for arm in ARMS:
        draft = AugmentedDrafter(target.config, cfg, cfg_state["drafter"]).cuda().eval()
        selected = torch.load(parent / f"best-{arm}.pt", weights_only=True)
        draft.load_state_dict(selected["drafter"])
        metrics[arm] = evaluate(
            draft,
            target,
            side_examples(examples, arm),
            cfg_state["mask_token_id"],
            include_examples=True,
        )
        del draft, selected
    baseline = AugmentedDrafter(target.config, cfg, cfg_state["drafter"]).cuda().eval()
    metrics["qualified_baseline"] = evaluate(
        baseline,
        target,
        side_examples(examples, "raw"),
        cfg_state["mask_token_id"],
        include_examples=True,
    )
    intervals = paired_intervals(metrics)
    result = {
        "completed": True,
        "frozen": frozen,
        "test_metrics": metrics,
        "paired_source_group_bootstrap": intervals,
        "test_exclusions": excluded,
        "benefit_criterion": "Predeclared primary J-minus-raw macro delta; positive lower paired 95% interval supports a gain on this held-out sample, one seed only",
        "target_gradients_absent": all(p.grad is None for p in target.parameters()),
        "test_status": "Fresh test consumed once after freezing; historical drafting test not evaluated",
        "training_parent": str(parent),
        "calibration_parent": calibration,
        "quality_note": "Target imitation; no speedup or unverified task-quality claim",
    }
    write_json(output / "report.json", result)
    commit()
    return {"completed": True, "output": str(output), "paired_comparison": intervals}
