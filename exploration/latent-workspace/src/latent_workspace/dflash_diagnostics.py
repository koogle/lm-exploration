"""Isolate verifier logic from GPU target arithmetic before further training."""

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .corpus import DOMAINS
from .dflash import DFlashDrafter, DraftConfig, verified_generate
from .dflash_train import evaluate, greedy_generate, precision
from .train import write_json


def replay_exposures(data, cfg, steps, expected_rng):
    """Count actual sampled conditioning positions; verify replay against saved RNG."""
    generator = torch.Generator().manual_seed(cfg.seed)
    pools = {d: [e for e in data["train"] if e["domain"] == d] for d in DOMAINS}
    context, prompt_context, blocks = 0, 0, 0
    for step in range(1, steps + 1):
        examples = []
        for i in range(cfg.batch_size):
            pool = pools[DOMAINS[((step - 1) * cfg.batch_size + i) % 3]]
            examples.append(pool[int(torch.randint(len(pool), (), generator=generator))])
        for e in examples:
            anchors = torch.randint(
                e["prompt_length"],
                len(e["tokens"]) - cfg.block_size + 1,
                (cfg.anchors_per_sequence,),
                generator=generator,
            )
            context += int(anchors.sum())
            prompt_context += e["prompt_length"] * len(anchors)
            blocks += len(anchors)
    if not torch.equal(generator.get_state(), expected_rng.cpu()):
        raise ValueError("sampling replay does not match saved RNG; exposure count invalid")
    return {
        "sampler_state_matches": True,
        "sampled_blocks": blocks,
        "supervised_new_token_exposures": blocks * (cfg.block_size - 1),
        "conditioning_position_exposures": context,
        "prompt_conditioning_position_exposures": prompt_context,
        "earlier_response_conditioning_position_exposures": context - prompt_context,
        "note": "Positions attended before each anchor, counted per sampled block. Cached features are reused; this is not backbone training or target forward FLOPs.",
    }


def audit_fit(parent, output):
    """Evaluate every training response at fixed anchors, without optimizer updates."""
    parent, output = Path(parent), Path(output)
    source_report = json.loads((parent / "report.json").read_text())
    if not source_report.get("training_completed"):
        raise ValueError("fit audit requires a finished training run")
    output.mkdir(parents=True, exist_ok=False)
    data = torch.load(parent / "teacher.pt", map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    for split, examples in data.items():
        for e in examples:
            digest.update(split.encode() + e["group"].encode() + e["tokens"].numpy().tobytes())
    if digest.hexdigest() != source_report["teacher_sequence_sha256"]:
        raise ValueError("teacher sequence identity mismatch")
    cfg = DraftConfig(**source_report["config"])
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model, revision=cfg.model_revision, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    draft = DFlashDrafter(target.config, cfg).cuda().eval()
    report = {
        "parent": str(parent),
        "teacher_sequence_sha256": digest.hexdigest(),
        "training_source_sha256": source_report["source_sha256"],
        "note": "All training and validation responses, up to eight fixed anchors per response; not every possible anchor. Same BF16 evaluation as training curves. No training or test access.",
        "checkpoints": {},
    }
    for name in ("best", "latest"):
        path = parent / f"{name}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        expected_step = (
            source_report["best_step"] if name == "best" else source_report["curve"][-1]["step"]
        )
        if state["step"] != expected_step:
            raise ValueError("checkpoint step does not match report")
        if name == "latest":
            report["sampling_exposures"] = replay_exposures(
                data, cfg, state["step"], state["sampler_rng"]
            )
        draft.load_state_dict(state["drafter"])
        with path.open("rb") as checkpoint_file:
            checkpoint_sha = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
        report["checkpoints"][name] = {
            "step": state["step"],
            "checkpoint_sha256": checkpoint_sha,
            "training": evaluate(draft, target, data["train"], state["mask_token_id"]),
            "validation": evaluate(draft, target, data["validation"], state["mask_token_id"]),
        }
        if name == "best":
            from .dflash import block_loss

            tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
            samples = []
            for domain in DOMAINS:
                e = next(x for x in data["validation"] if x["domain"] == domain)
                anchors = sorted(
                    {
                        e["prompt_length"],
                        min(e["prompt_length"] + 16, len(e["tokens"]) - cfg.block_size),
                    }
                )
                for anchor in anchors:
                    with torch.no_grad(), precision("cuda"):
                        _, _, predicted, labels = block_loss(
                            draft,
                            target,
                            e["tokens"][None].cuda(),
                            e["memory"][None].cuda(),
                            torch.tensor([[anchor]], device="cuda"),
                            state["mask_token_id"],
                        )
                    samples.append(
                        {
                            "domain": domain,
                            "group": e["group"],
                            "anchor_position": anchor,
                            "context_tail": tokenizer.decode(
                                e["tokens"][max(0, anchor - 48) : anchor]
                            ),
                            "supplied_anchor": tokenizer.decode(e["tokens"][anchor : anchor + 1]),
                            "teacher_new_tokens": labels[0].tolist(),
                            "draft_new_tokens": predicted[0].tolist(),
                            "teacher_text": tokenizer.decode(labels[0]),
                            "draft_text": tokenizer.decode(predicted[0]),
                            "consecutive_new_matches": int(
                                (predicted == labels).long().cumprod(-1).sum()
                            ),
                        }
                    )
            report["qualitative_blocks"] = samples
        write_json(output / "fit-audit.json", report)
        print(f"Full-response fitting audit: {name} step {state['step']}", flush=True)
    return report


@torch.no_grad()
def teacher_forced_logits(target, prompt, response, chunk_size):
    output = target(prompt, use_cache=True, logits_to_keep=1)
    logits = [output.logits[:, -1:].float().cpu()]
    for chunk in torch.tensor([response[:-1]], device=prompt.device).split(chunk_size, dim=1):
        if not chunk.numel():
            continue
        output = target(chunk, past_key_values=output.past_key_values, use_cache=True)
        logits.append(output.logits.float().cpu())
    return torch.cat(logits, dim=1)[0]


def diagnose(parent, output):
    parent, output = Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    state = torch.load(parent / "best.pt", map_location="cpu", weights_only=True)
    cfg = DraftConfig(**state["config"])
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model, revision=cfg.model_revision, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    draft = DFlashDrafter(target.config, cfg).cuda().eval()
    draft.load_state_dict(state["drafter"])
    data = torch.load(parent / "teacher.pt", map_location="cpu", weights_only=True)
    examples = [next(e for e in data["validation"] if e["domain"] == d) for d in DOMAINS]
    report = {
        "parent": str(parent),
        "checkpoint_step": state["step"],
        "cases": [],
        "note": "Same stored target weights; FP32 mode upcasts BF16 weights. No training, no test evaluation. Timing is diagnostic only.",
    }
    for dtype in ("bfloat16", "float32"):
        if dtype == "float32":
            target.float()
        for example in examples:
            prompt = example["tokens"][: example["prompt_length"]][None].cuda()
            with precision("cuda") if dtype == "bfloat16" else nullcontext():
                ar = greedy_generate(target, prompt, 32, tokenizer.eos_token_id)
                spec = verified_generate(
                    draft, target, prompt, 32, state["mask_token_id"], tokenizer.eos_token_id
                )
                sequential = teacher_forced_logits(target, prompt, ar["tokens"], 1)
                blocked = teacher_forced_logits(target, prompt, ar["tokens"], cfg.block_size)
            indices = (sequential.argmax(-1) != blocked.argmax(-1)).nonzero().flatten().tolist()
            differences = []
            for i in indices:
                top = sequential[i].topk(2)
                differences.append(
                    {
                        "position": i,
                        "sequential_argmax": int(sequential[i].argmax()),
                        "block_argmax": int(blocked[i].argmax()),
                        "sequential_top2_gap": float(top.values[0] - top.values[1]),
                        "maximum_logit_difference": float((sequential[i] - blocked[i]).abs().max()),
                    }
                )
            case = {
                "domain": example["domain"],
                "target_arithmetic": dtype,
                "ar_tokens": ar["tokens"],
                "speculative_tokens": spec["tokens"],
                "verified_matches_ar": ar["tokens"] == spec["tokens"],
                "fixed_input_chunk_argmax_differences": differences,
                "ar_seconds": ar["total_seconds"],
                "speculative_seconds": spec["total_seconds"],
                "accepted_beyond_anchor": spec["mean_accepted_beyond_anchor"],
            }
            report["cases"].append(case)
            write_json(output / "diagnostics.json", report)
            print(
                f"{dtype} {example['domain']}: match={case['verified_matches_ar']} fixed-input flips={len(indices)}",
                flush=True,
            )
    return report


@torch.no_grad()
def audit_teacher(parent, output):
    """Small train/validation consistency audit; no optimizer or test evaluation."""
    from .dflash import features

    parent, output = Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = DraftConfig(**json.loads((parent / "config.json").read_text()))
    data = torch.load(parent / "teacher.pt", map_location="cpu", weights_only=True)
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model,
            revision=cfg.model_revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    cases = []
    for split in ("train", "validation"):
        for domain in DOMAINS:
            e = next(x for x in data[split] if x["domain"] == domain)
            tokens = e["tokens"][None].cuda()
            result = target(tokens, use_cache=False, output_hidden_states=True)
            start = e["prompt_length"]
            predicted = result.logits[:, start - 1 : -1].argmax(-1)
            expected = tokens[:, start:]
            memory = features(result, cfg.layers)[0].float().cpu()
            cached = e["memory"].float()
            cosine = torch.nn.functional.cosine_similarity(memory, cached, dim=-1).mean()
            changed = tokens.clone()
            changed[:, start:] = (changed[:, start:] + 1) % target.config.vocab_size
            altered = target.model(changed, use_cache=False, output_hidden_states=True)
            prefix_delta = (
                (features(altered, cfg.layers)[0, :start].float().cpu() - memory[:start])
                .abs()
                .max()
            )
            cases.append(
                {
                    "split": split,
                    "domain": domain,
                    "group": e["group"],
                    "response_tokens": expected.numel(),
                    "full_forward_teacher_argmax_agreement": float(
                        (predicted == expected).float().mean()
                    ),
                    "cached_feature_mean_cosine": float(cosine),
                    "cached_feature_relative_rms_difference": float(
                        (cached - memory).square().mean().sqrt() / memory.square().mean().sqrt()
                    ),
                    "future_perturbation_prefix_max_difference": float(prefix_delta),
                }
            )
    report = {
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "parent": str(parent),
        "cases": cases,
        "completed": True,
        "scope": "six source examples only; BF16 batch-shape differences permitted; not an exhaustive cache audit",
        "test_status": "not accessed",
    }
    write_json(output / "report.json", report)
    return report
