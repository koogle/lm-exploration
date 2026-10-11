"""Data-size/optimization learning curves; test metrics stay sealed."""

import copy
import hashlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch

from .backbone import FrozenBackbone
from .corpus import DOMAINS, audit_windows, extend_frozen_corpus, prepare
from .evaluate import evaluate, generate, repeated_next_token_baseline, teacher_baseline
from .models import BlockHead, loss_and_logits
from .scan import cached_metrics
from .train import atomic_save, checkpoint, seed_all, write_json


def nested_indices(ranges, sizes, seed, priority=None):
    generator = torch.Generator().manual_seed(seed)
    orders = {
        domain: torch.randperm(end - start, generator=generator) + start
        for domain, (start, end) in ranges.items()
    }
    if priority:
        for domain, first in priority.items():
            first = torch.tensor(first, dtype=torch.long)
            orders[domain] = torch.cat([first, orders[domain][~torch.isin(orders[domain], first)]])
    return {
        size: torch.cat([orders[domain][: size // len(DOMAINS)] for domain in DOMAINS])
        for size in sizes
    }


def reference_diagnostics(path, backbone, cfg, device):
    from .config import Config

    path = Path(path)
    report = json.loads((path / "study.json").read_text())
    for key in ("model_revision", "context_length", "block_size", "memory_tokens", "layers"):
        if report["config"][key] != json.loads(json.dumps(cfg.to_dict()))[key]:
            raise ValueError(f"reference incompatible: {key}")
    corpus = torch.load(path / "corpus.pt", weights_only=True)["tensors"]["train"]
    metrics = {}
    for size, curve in report["curves"].items():
        if int(size) > 256:
            raise ValueError("reference diagnostics are bounded to tiny training sets")
        state = torch.load(path / f"size-{size}.pt", weights_only=True, map_location="cpu")
        reference_cfg = Config(**state["config"])
        head = BlockHead(backbone.hidden_size, reference_cfg, state["variant"]).to(device)
        head.load_state_dict(state["head"])
        rows = corpus[curve["train_indices"]]
        memory = torch.cat(
            [
                backbone.encode(batch[:, : cfg.context_length].to(device))[0].cpu()
                for batch in rows.split(cfg.batch_size)
            ]
        )
        metrics[size] = cached_metrics(
            head, backbone, memory, rows[:, cfg.context_length :], cfg, device
        )
    return {"run": str(path), "training_only": metrics}


def run_study(cfg, output, on_checkpoint=None):
    cfg.validate()
    if cfg.corpus != "mixed" or len(cfg.variants) != 1 or cfg.evaluate_test:
        raise ValueError(
            "data study requires mixed training, one fixed variant and a sealed test set"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "study.json").exists():
        raise ValueError("study already exists; use a new run ID")
    start = time.monotonic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(cfg.seed)
    parent = None
    tokenizer_compatibility = None
    if cfg.study_parent:
        parent_path = Path(cfg.study_parent)
        parent = json.loads((parent_path / "study.json").read_text())
        for key in ("context_length", "block_size"):
            if cfg.to_dict()[key] != parent["config"][key]:
                raise ValueError(f"parent corpus incompatible: {key}")
        if cfg.model == parent["config"]["model"]:
            cfg.model_revision = parent["config"]["model_revision"]
        elif not cfg.allow_compatible_backbone_change:
            raise ValueError("parent model differs; explicit compatible-backbone reuse required")
    backbone = FrozenBackbone(cfg, device)
    cfg.model_revision = backbone.revision
    if parent and cfg.model != parent["config"]["model"]:
        from transformers import AutoTokenizer

        from .backbone import compatible_tokenizers

        source_tokenizer = AutoTokenizer.from_pretrained(
            parent["config"]["model"], revision=parent["config"]["model_revision"]
        )
        tokenizer_compatibility = compatible_tokenizers(source_tokenizer, backbone.tokenizer)
        tokenizer_compatibility["parent_model"] = parent["config"]["model"]
        tokenizer_compatibility["parent_revision"] = parent["config"]["model_revision"]
    priority = None
    if parent:
        base = torch.load(parent_path / "corpus.pt", weights_only=True)
        if cfg.train_examples == len(base["tensors"]["train"]):
            data, metadata = base["tensors"], base["metadata"]
            maps = {
                domain: {i: i for i in range(a, b)}
                for domain, (a, b) in metadata["domain_ranges"]["train"].items()
            }
        else:
            candidate_cfg = replace(
                cfg, train_examples=cfg.train_examples + max(96, cfg.train_examples // 96 * 3)
            )
            candidate = prepare(candidate_cfg, backbone.tokenizer, base["metadata"])
            atomic_save(
                output / "candidate.pt", {"tensors": candidate[0], "metadata": candidate[1]}
            )
            data, metadata, maps = extend_frozen_corpus(base, candidate, cfg.train_examples)
        original_order = parent["curves"][str(len(base["tensors"]["train"]))]["train_indices"]
        priority = {}
        for domain, mapping in maps.items():
            priority[domain] = list(
                dict.fromkeys(mapping[i] for i in original_order if i in mapping)
            )
        for split in ("validation", "test"):
            if len(data[split]) != getattr(
                cfg, "eval_examples" if split == "validation" else "test_examples"
            ):
                raise ValueError("parent held-out sizes must be preserved")
    else:
        data, metadata = prepare(cfg, backbone.tokenizer)
    window_audit = audit_windows(data)
    atomic_save(output / "corpus.pt", {"tensors": data, "metadata": metadata})
    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(source.name.encode() + source.read_bytes())
    result = {
        "config": cfg.to_dict(),
        "data": metadata,
        "window_audit": window_audit,
        "tokenizer_compatibility": tokenizer_compatibility,
        "source_sha256": source_hash.hexdigest(),
        "test_status": "no test losses or accuracy computed in this run",
        "curves": {},
        "readout_norm_weight_rms": float(
            backbone.model.model.norm.weight.float().square().mean().sqrt()
        ),
    }

    result["teacher_validation_by_domain"] = {
        domain: teacher_baseline(backbone, data["validation"][a:b], cfg, device)
        for domain, (a, b) in metadata["domain_ranges"]["validation"].items()
    }

    result["repeated_next_token_baseline"] = {
        domain: repeated_next_token_baseline(backbone, data["validation"][a:b], cfg, device)
        for domain, (a, b) in metadata["domain_ranges"]["validation"].items()
    }

    if cfg.study_reference:
        result["reference_diagnostics"] = reference_diagnostics(
            cfg.study_reference, backbone, cfg, device
        )

    def persist():
        write_json(output / "study.json", result)
        if on_checkpoint:
            on_checkpoint()

    persist()
    if any(v.endswith(":jspace") for v in cfg.variants):
        from .jspace import fit_jacobians

        eligible = (data["train"][:, : cfg.context_length].ge(0).sum(-1) >= 18).nonzero().flatten()
        order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(cfg.seed + 1))
        indices = eligible[order[: cfg.jspace_prompts]]
        if len(indices) != cfg.jspace_prompts:
            raise ValueError("not enough eligible training prefixes for J-lens calibration")
        result["calibration_indices"] = indices.tolist()
        backbone.jacobians = fit_jacobians(
            backbone,
            data["train"][indices, : cfg.context_length],
            deadline=start + cfg.max_seconds,
            checkpoint_path=output / "jacobian-calibration.pt",
            on_checkpoint=on_checkpoint,
        )
        atomic_save(output / "jacobians.pt", backbone.jacobians)
        backbone.conditioning = "jspace"
        persist()
    subsets = nested_indices(
        metadata["domain_ranges"]["train"], cfg.study_sizes, cfg.seed, priority
    )
    cache_indices = (
        torch.arange(len(data["train"])) if cfg.study_scan_only else subsets[max(subsets)]
    )
    lookup = torch.full((len(data["train"]),), -1, dtype=torch.long)
    lookup[cache_indices] = torch.arange(len(cache_indices))
    memories = {}
    for split in ("train", "validation"):
        destination = None
        offset = 0
        cache_rows = data[split][cache_indices] if split == "train" else data[split]
        for rows in cache_rows.split(cfg.batch_size):
            features = backbone.encode(rows[:, : cfg.context_length].to(device))[0].cpu()
            if not cfg.scan_attention and cfg.variants[0].split(":")[1] in (
                "multi",
                "final",
                "all",
                "best",
            ):
                features = features[:, : len(cfg.layers)]
            if destination is None:
                destination = torch.empty(
                    (len(cache_rows), *features.shape[1:]), dtype=features.dtype
                )
            destination[offset : offset + len(rows)].copy_(features)
            offset += len(rows)
            if time.monotonic() - start > cfg.max_seconds:
                raise TimeoutError("data study budget exhausted extracting features")
        memories[split] = destination
    if cfg.study_scan_only:
        from .scan import layer_scan

        if cfg.scan_steps <= 0 or cfg.residual_readout:
            raise ValueError(
                "information scan needs positive steps and no final-state residual shortcut"
            )

        def save_scan(layers):
            write_json(output / "layer-scan-progress.json", layers)
            if on_checkpoint:
                on_checkpoint()

        result["layer_scan"] = layer_scan(
            backbone,
            memories["train"],
            data["train"][:, cfg.context_length :],
            memories["validation"],
            data["validation"][:, cfg.context_length :],
            cfg,
            device,
            start + cfg.max_seconds,
            domain_ranges=metadata["domain_ranges"]["validation"],
            on_layer=save_scan,
        )
        result["completed"] = True
        result["elapsed_seconds"] = time.monotonic() - start
        persist()
        return result
    variant = cfg.variants[0]
    for size, subset in subsets.items():
        seed_all(cfg.seed)
        head = BlockHead(backbone.hidden_size, cfg, variant).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
        sampler = torch.Generator().manual_seed(cfg.seed)
        selected = data["train"][subset]
        memory = memories["train"]
        per_domain = size // len(DOMAINS)
        diagnostic_indices = torch.cat(
            [
                torch.arange(
                    i * per_domain,
                    i * per_domain + min(cfg.study_train_diagnostics // len(DOMAINS), per_domain),
                )
                for i in range(len(DOMAINS))
            ]
        )
        if len(torch.unique(subset)) != size:
            raise ValueError("duplicate subset indices")
        distinct_windows = size
        masked_exposures = 0
        curve = {
            "training_windows": size,
            "trainable_parameters": sum(p.numel() for p in head.parameters() if p.requires_grad),
            "unique_blocks": distinct_windows,
            "unique_target_token_slots": distinct_windows * cfg.block_size,
            "train_indices": subset.tolist(),
            "checkpoints": [],
        }
        start_step = 0
        if cfg.study_resume:
            resume_path = Path(cfg.study_resume)
            previous = json.loads((resume_path / "study.json").read_text())
            previous_curve = previous["curves"][str(size)]
            state = torch.load(
                resume_path / f"size-{size}.pt", weights_only=True, map_location="cpu"
            )
            from .config import Config

            old_cfg = Config(**state["config"])
            critical = (
                "model",
                "model_revision",
                "layers",
                "selected_layers",
                "block_size",
                "context_length",
                "memory_tokens",
                "slots",
                "width",
                "heads",
                "decoder_layers",
                "refinement_rounds",
                "diffusion_steps",
                "batch_size",
                "learning_rate",
                "seed",
                "variants",
                "position_readout",
                "residual_readout",
                "layer_dropout",
                "null_attention",
                "normalize_token_inputs",
            )
            for key in critical:
                if json.dumps(old_cfg.to_dict()[key]) != json.dumps(cfg.to_dict()[key]):
                    raise ValueError(f"resume changes training configuration: {key}")
            old_corpus = torch.load(resume_path / "corpus.pt", weights_only=True)
            old_rows = old_corpus["tensors"]["train"][previous_curve["train_indices"]]
            if not torch.equal(selected, old_rows):
                raise ValueError("resume changes ordered training examples")
            for split in ("validation", "test"):
                if not torch.equal(data[split], old_corpus["tensors"][split]):
                    raise ValueError("resume changes held-out tensors")
            start_step = state["step"]
            if max(cfg.study_checkpoints) <= start_step:
                raise ValueError("resume must extend the update budget")
            if previous_curve["checkpoints"][-1]["step"] != start_step:
                raise ValueError("resume checkpoint/report step mismatch")
            head.load_state_dict(state["head"])
            optimizer.load_state_dict(state["optimizer"])
            sampler.set_state(state["sampler_rng"])
            torch.set_rng_state(state["rng"])
            if state["cuda_rng"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            curve = copy.deepcopy(previous_curve)
            curve["resume_source"] = str(resume_path)
            curve["resume_parent_source_sha256"] = previous["source_sha256"]
            masked_exposures = curve["checkpoints"][-1]["masked_supervised_tokens_seen"]
            best_path = resume_path / f"size-{size}-best.pt"
            if best_path.exists():
                shutil.copyfile(best_path, output / best_path.name)
            del old_corpus, old_rows, state
        result["curves"][str(size)] = curve
        for step in range(start_step + 1, max(cfg.study_checkpoints) + 1):
            if time.monotonic() - start > cfg.max_seconds:
                result["stopped"] = "time budget; incomplete curves"
                persist()
                return result
            head.train()
            if hasattr(head, "workspace"):
                head.workspace.dropout_step = step
            index = torch.randint(size, (cfg.batch_size,), generator=sampler)
            optimizer.zero_grad(set_to_none=True)
            loss, _, mask = loss_and_logits(
                head,
                backbone,
                memory[lookup[subset[index]]].to(device),
                selected[index, cfg.context_length :].to(device),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite study loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            masked_exposures += int(mask.sum())
            if step in cfg.study_checkpoints:
                train_metric = cached_metrics(
                    head,
                    backbone,
                    memory[lookup[subset[diagnostic_indices]]],
                    selected[diagnostic_indices, cfg.context_length :],
                    cfg,
                    device,
                )
                by_domain = {}
                for domain, (a, b) in metadata["domain_ranges"]["validation"].items():
                    by_domain[domain] = evaluate(
                        head, backbone, data["validation"][a:b], cfg, device
                    )
                row = {
                    "step": step,
                    "expected_training_passes": step * cfg.batch_size / size,
                    "training_diagnostic_windows": len(diagnostic_indices),
                    "target_token_slots_seen": step * cfg.batch_size * cfg.block_size,
                    "masked_supervised_tokens_seen": masked_exposures,
                    "training_sample_nll_by_position": train_metric["nll_by_position"],
                    "training_sample_accuracy_by_position": train_metric["accuracy_by_position"],
                    "training_position_collapse_fraction": train_metric[
                        "same_prediction_at_all_positions"
                    ],
                    "validation_by_domain": by_domain,
                    "macro_validation_nll": sum(v["nll"] for v in by_domain.values())
                    / len(DOMAINS),
                }
                curve["checkpoints"].append(row)
                if row["macro_validation_nll"] < curve.get("best_validation_nll", float("inf")):
                    curve["best_validation_nll"] = row["macro_validation_nll"]
                    curve["best_step"] = step
                    atomic_save(
                        output / f"size-{size}-best.pt",
                        {
                            "head": head.state_dict(),
                            "jacobians": backbone.jacobians,
                            "step": step,
                            "config": cfg.to_dict(),
                            "variant": head.variant,
                            "model_revision": backbone.revision,
                            "selection": "minimum macro validation NLL; no test access",
                        },
                    )
                checkpoint(
                    output / f"size-{size}.pt", head, optimizer, step, cfg, backbone, sampler
                )
                result["elapsed_seconds"] = time.monotonic() - start
                persist()
                print(
                    f"Data study size={size} step={step} validation_nll={row['macro_validation_nll']:.4f} supervised_tokens={masked_exposures}",
                    flush=True,
                )
        curve["validation_samples"] = {}
        for domain, (a, _) in metadata["domain_ranges"]["validation"].items():
            prefix = data["validation"][a : a + 1, : cfg.context_length].to(device)
            tokens = generate(head, backbone, prefix, max_new_tokens=24)
            curve["validation_samples"][domain] = {
                "prefix": backbone.tokenizer.decode(prefix[0][prefix[0] >= 0].tolist()),
                "generated": backbone.tokenizer.decode(tokens),
            }
        persist()
        del head, optimizer, memory
    result["completed"] = True
    result["elapsed_seconds"] = time.monotonic() - start
    persist()
    return result
