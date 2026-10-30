"""Render a Markdown model-card document using verified measured reports."""

import argparse
import json
import math
import re
import shutil
import tomllib
from pathlib import Path
from string import Template

DEFAULT_DOCUMENT = Path(__file__).with_name("cards") / "completed.md"


def load_document(path):
    text = Path(path).read_text()
    if not text.startswith("+++\n"):
        raise ValueError("Card document requires TOML front matter delimited by +++")
    try:
        metadata, markdown = text[4:].split("\n+++\n", 1)
    except ValueError as exc:
        raise ValueError("Unclosed card document front matter") from exc
    settings = tomllib.loads(metadata)
    if not settings.get("segments"):
        raise ValueError("Card document must reference at least one training segment")
    count = settings.get("displayed_samples")
    if type(count) is not int or count < 0:
        raise ValueError("displayed_samples must be a nonnegative integer")
    return settings, Template(markdown)


def sample_markdown(sample):
    # A generated completion can itself contain Markdown code fences.
    completion = sample["completion"]
    fence = "`" * max(3, 1 + max((len(s) for s in re.findall(r"`+", completion)), default=0))
    return (
        f"Prompt: {sample['prompt']}\n\nMode: {sample['mode']}; seed: {sample['seed']}\n\n"
        f"{fence}text\n{completion}\n{fence}"
    )


def card_values(settings, audit, validation, segments):
    for key in (
        "strict_load",
        "per_source_two_pass_limit_verified",
        "finite_parameters",
        "token_accounting_verified",
    ):
        if validation.get(key) is not True:
            raise ValueError(f"Validation did not verify {key}")
    for key in (
        "all_metrics_finite",
        "contiguous_steps",
        "optimizer_states_finite",
        "optimizer_step_counters_verified",
        "session_resume_validation_boundaries_exact",
    ):
        if audit.get(key) is not True:
            raise ValueError(f"Training audit did not verify {key}")
    final = segments[-1]
    config = final["config"]
    tokens, updates = audit["tokens"], audit["updates"]
    if not tokens == validation["trained_tokens"] == final["tokens"]:
        raise ValueError("Training token counts disagree across reports")
    if not updates == validation["step"] == final["steps"]:
        raise ValueError("Training update counts disagree across reports")
    if validation["parameters"] != final["parameters"]:
        raise ValueError("Parameter counts disagree across reports")
    if not (
        audit["data_manifest_sha256"]
        == validation["data_manifest_sha256"]
        == final["data_identity"]["manifest_sha256"]
    ):
        raise ValueError("Dataset identities disagree across reports")
    if sum(audit["source_exposures"].values()) != tokens:
        raise ValueError("Source exposures do not sum to trained tokens")
    trained = validation["trained"]
    baseline = validation[settings["baseline"]]
    if {name: row["tokens"] for name, row in trained["per_source"].items()} != {
        name: row["tokens"] for name, row in baseline["per_source"].items()
    }:
        raise ValueError("Final and baseline evaluation targets disagree")
    count = settings["displayed_samples"]
    if count > len(validation["samples"]):
        raise ValueError("Requested more displayed samples than the report contains")
    local_batch = config["world_size"] * config["microbatch"]
    if config["global_windows"] % local_batch:
        raise ValueError("Training batch does not divide into whole accumulation rounds")
    if config["n_embd"] % config["n_head"]:
        raise ValueError("Model width does not divide into whole attention heads")
    values = {
        **settings.get("historical_training", {}),
        **config,
        "parameters": f"{validation['parameters']:,}",
        "trained_tokens": f"{tokens:,}",
        "updates": f"{updates:,}",
        "target_tokens": f"{config['target_tokens']:,}",
        "unused_tokens": f"{config['target_tokens'] - tokens:,}",
        "layers": config["n_layer"],
        "width": config["n_embd"],
        "heads": config["n_head"],
        "head_dimension": config["n_embd"] // config["n_head"],
        "context": f"{config['context']:,}",
        "world_size": config["world_size"],
        "gpu": final["gpu"],
        "training_torch_version": final["torch_version"],
        "seed": config["seed"],
        "weight_decay": config["weight_decay"],
        "global_windows": config["global_windows"],
        "microbatch": config["microbatch"],
        "accumulation_rounds": config["global_windows"] // local_batch,
        "warmup_tokens": f"{config['warmup_tokens']:,}",
        "peak_lr": config["peak_lr"],
        "min_lr_percent": f"{config['min_lr_ratio'] * 100:g}",
        "session_hours": f"{audit['total_session_seconds'] / 3600:.2f}",
        "tokens_per_second": f"{tokens / audit['total_session_seconds']:,.0f}",
        "source_shares": ", ".join(
            f"{name} {exposures / tokens:.3%}"
            for name, exposures in audit["source_exposures"].items()
        ),
        "unique_train_tokens": f"{sum(s['unique_train_tokens'] for s in validation['source_exposures'].values()):,}",
        "evaluation_policy": validation["evaluation"],
        "evaluation_parents": f"{sum(r['parents'] for r in trained['per_source'].values()):,}",
        "evaluation_targets": f"{sum(r['tokens'] for r in trained['per_source'].values()):,}",
        "evaluation_device": validation["evaluation_device"].upper(),
        "evaluation_torch_version": validation["torch_version"],
        "baseline_label": settings["baseline_label"],
        "baseline_loss": f"{baseline['mixture_weighted_loss']:.4f}",
        "baseline_perplexity": f"{math.exp(baseline['mixture_weighted_loss']):.2f}",
        "trained_loss": f"{trained['mixture_weighted_loss']:.4f}",
        "trained_perplexity": f"{math.exp(trained['mixture_weighted_loss']):.2f}",
        "source_rows": "\n".join(
            f"| {name} | {r['tokens']:,} | {r['loss']:.4f} | {r['perplexity']:.2f} |"
            for name, r in trained["per_source"].items()
        ),
        "diagnostic_loss": f"{final['final_validation']['weighted_loss']:.4f}",
        "pending_audits": "; ".join(validation["pending_data_audits"]) or "none recorded",
        "prompt_count": len({s["prompt"] for s in validation["samples"]}),
        "completion_count": len(validation["samples"]),
        "displayed_samples": count,
        "sample_text": "\n\n".join(sample_markdown(s) for s in validation["samples"][:count]),
        "segment_count": len(segments),
        "final_checkpoint": final["final_checkpoint"],
    }
    return values


def build_card(artifacts, output, document=DEFAULT_DOCUMENT):
    settings, template = load_document(document)
    names = [settings["audit"], settings["validation"], *settings["segments"]]
    paths = [Path(artifacts) / name for name in names]
    audit, validation, *segments = [json.loads(path.read_text()) for path in paths]
    values = card_values(settings, audit, validation, segments)
    # Fail on unknown placeholders before writing any output; never evaluate code.
    markdown = template.substitute(values)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(markdown)
    for source, name in zip(
        paths,
        [
            "training-audit.json",
            "validation.json",
            *[f"training-segment-{i:02d}.json" for i in range(len(segments))],
        ],
    ):
        shutil.copyfile(source, output / name)
    return {
        "evaluation_targets": sum(
            r["tokens"] for r in validation["trained"]["per_source"].values()
        ),
        "final_loss": validation["trained"]["mixture_weighted_loss"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    args = parser.parse_args()
    print(build_card(args.artifacts, args.output, args.document))


if __name__ == "__main__":
    main()
