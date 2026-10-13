"""Read-only qualification of one frozen draft checkpoint; never trains or selects on test."""

import hashlib
import json
import time
from pathlib import Path

import torch

from latent_workspace.corpus import DOMAINS
from latent_workspace.dflash import DFlashDrafter, DraftConfig, verified_generate
from latent_workspace.dflash_data import load_teacher_cache, token_hash
from latent_workspace.dflash_scaling import generate_batch
from latent_workspace.dflash_train import evaluate, greedy_generate
from latent_workspace.train import atomic_save, write_json


def gate(metrics):
    return metrics["accepted_beyond_anchor"] >= 2 and all(
        metrics["by_domain"][d]["mean_accepted_beyond_anchor"] >= 1 for d in DOMAINS
    )


def require_parity(rows):
    if not rows or not all(r["identical_to_target_greedy"] for r in rows):
        raise ValueError("Long-rollout parity failed; sealed test must remain unopened")


def audit_test_prompts(records):
    test = [r for r in records if r["split"] == "test"]
    other = [r for r in records if r["split"] != "test"]
    groups = {r.get("source_group", r["group"]) for r in other}
    prefixes = {token_hash(r["prompt_ids"]) for r in other}
    seen = set()
    for r in test:
        if r.get("source_group", r["group"]) in groups or token_hash(r["prompt_ids"]) in prefixes:
            raise ValueError("Sealed test overlaps training/validation source or exact prompt")
        if r["group"] in seen:
            raise ValueError("Duplicate test identity")
        seen.add(r["group"])
    if any(sum(r["domain"] == d for r in test) != 64 for d in DOMAINS):
        raise ValueError("Expected 64 reserved test prompts per domain")
    return test


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def qualify(protocol, output, commit):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(output)
    output.mkdir(exist_ok=False)
    started = time.monotonic()

    def remaining():
        if time.monotonic() - started > protocol["max_seconds"] - 90:
            raise TimeoutError("Bounded qualification deadline; frozen artifacts preserved")

    parent = Path(protocol["checkpoint_parent"])
    state = torch.load(parent / "best.pt", map_location="cpu", weights_only=True)
    training = json.loads((parent / "report.json").read_text())
    if (
        state["step"] != protocol["checkpoint_step"]
        or state["source_sha256"] != protocol["training_source_sha256"]
    ):
        raise ValueError("Selected checkpoint provenance changed")
    cfg = DraftConfig(**state["config"]).validate()
    for key in ("model", "model_revision", "layers", "draft_layers", "qk_norm", "block_size"):
        if json.dumps(cfg.to_dict()[key]) != json.dumps(protocol[key]):
            raise ValueError(f"Frozen architecture changed: {key}")
    if not gate(training["selected_validation"]):
        raise ValueError("Selected checkpoint fails predeclared validation gate")
    frozen = {
        **protocol,
        "checkpoint_sha256": file_sha(parent / "best.pt"),
        "training_report_sha256": file_sha(parent / "report.json"),
        "prompt_manifest_sha256": file_sha(Path(protocol["manifest_parent"]) / "prompts.json"),
        "evaluator_sha256": file_sha(__file__),
        "selected_validation": training["selected_validation"],
        "test_status": "UNOPENED: checkpoint and evaluation protocol frozen",
    }
    write_json(output / "frozen.json", frozen)
    commit()
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
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    draft = DFlashDrafter(target.config, cfg).cuda().eval()
    draft.load_state_dict(state["drafter"])
    del state
    mask_id = training["mask_token_id"]
    validation = load_teacher_cache(parent / "teacher.pt")["validation"]
    selected = [
        e
        for d in DOMAINS
        for e in [x for x in validation if x["domain"] == d][: protocol["rollouts_per_domain"]]
    ]
    rows = []
    target.float()
    with torch.inference_mode():
        for e in selected:
            remaining()
            prompt = e["tokens"][: e["prompt_length"]][None].cuda()
            ar = greedy_generate(target, prompt, protocol["rollout_tokens"], tokenizer.eos_token_id)
            spec = verified_generate(
                draft, target, prompt, protocol["rollout_tokens"], mask_id, tokenizer.eos_token_id
            )
            response = ar["tokens"]
            unique = len(
                {tuple(response[i : i + 4]) for i in range(max(0, len(response) - 3))}
            ) / max(1, len(response) - 3)
            rows.append(
                {
                    "domain": e["domain"],
                    "group": e["group"],
                    "output_tokens": len(response),
                    "identical_to_target_greedy": response == spec["tokens"],
                    "accepted_beyond_anchor": spec["mean_accepted_beyond_anchor"],
                    "accepted_new_draft_tokens": spec["accepted_new_draft_tokens"],
                    "verification_calls": spec["verification_calls"],
                    "target_tokens": response,
                    "draft_verified_tokens": spec["tokens"],
                    "target_text": tokenizer.decode(response),
                    "unique_response_4gram_fraction": unique,
                    "ar_seconds": ar["total_seconds"],
                    "speculative_seconds": spec["total_seconds"],
                }
            )
            write_json(
                output / "long-rollouts.json",
                {
                    "rows": rows,
                    "target_arithmetic": "float32 on stored BF16 weights",
                    "test_status": "UNOPENED",
                },
            )
            commit()
            print(
                f"Long rollout {len(rows)}/{len(selected)} {e['domain']} tokens={len(response)} parity={rows[-1]['identical_to_target_greedy']}",
                flush=True,
            )
    require_parity(rows)
    # Test labels are first generated only after the frozen checkpoint passes parity.
    manifest = json.loads((Path(protocol["manifest_parent"]) / "prompts.json").read_text())
    test_records = audit_test_prompts(manifest["records"])
    write_json(
        output / "test-opened.json",
        {
            "frozen_checkpoint_sha256": frozen["checkpoint_sha256"],
            "evaluation": "one frozen evaluation; no retuning",
            "reserved_prompts": len(test_records),
        },
    )
    commit()
    target.bfloat16()
    progress = json.loads((Path(protocol["teacher_parent"]) / "progress.json").read_text())
    seen = set(progress["response_hashes"])
    exclusions, examples = [], []
    with torch.inference_mode():
        for offset in range(0, len(test_records), protocol["teacher_batch_size"]):
            remaining()
            batch = test_records[offset : offset + protocol["teacher_batch_size"]]
            sequences = generate_batch(
                target, batch, protocol["response_tokens"], tokenizer.eos_token_id
            )
            for record, sequence in zip(batch, sequences):
                response = sequence[len(record["prompt_ids"]) :]
                sha = token_hash(response)
                if len(response) < cfg.block_size or sha in seen:
                    exclusions.append(
                        {
                            "group": record["group"],
                            "domain": record["domain"],
                            "reason": "short"
                            if len(response) < cfg.block_size
                            else "exact_response_duplicate",
                        }
                    )
                    continue
                seen.add(sha)
                examples.append(
                    {
                        **record,
                        "tokens": torch.tensor(sequence),
                        "prompt_length": len(record["prompt_ids"]),
                        "response_sha256": sha,
                    }
                )
            atomic_save(output / "test-teacher.pt", {"test": examples})
            write_json(
                output / "test-generation.json",
                {
                    "processed": min(offset + len(batch), len(test_records)),
                    "exclusions": exclusions,
                    "usable": len(examples),
                },
            )
            commit()
        if any(sum(e["domain"] == d for e in examples) < 2 for d in DOMAINS):
            raise ValueError("Insufficient usable test responses in a domain")
        remaining()
        metrics = evaluate(draft, target, examples, mask_id, include_examples=True)
        sources = {e["group"]: e.get("source_group", e["group"]) for e in examples}
        for row in metrics["examples"]:
            row["source_group"] = sources[row["group"]]
    report = {
        "completed": True,
        "frozen": frozen,
        "long_rollouts": rows,
        "long_rollout_parity_passed": True,
        "evaluation_split": "test",
        "test_metrics": metrics,
        "test_gate_passed": gate(metrics),
        "test_reserved_prompts": len(test_records),
        "test_usable_prompts": len(examples),
        "test_exclusions": exclusions,
        "elapsed_seconds": time.monotonic() - started,
        "frozen_target_has_no_gradients": all(p.grad is None for p in target.parameters()),
        "limitations": "One selected checkpoint/seed. 24 validation rollouts, maximum 256 tokens; EOS may end earlier. No test-based retuning. Cached teacher metrics use BF16; verified rollouts use FP32. Timing is one run per method, not a speed benchmark.",
    }
    write_json(output / "report.json", report)
    commit()
    return report
