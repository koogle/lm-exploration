"""Summarize one completed run without external dependencies or significance claims."""

import argparse
import json
from pathlib import Path


def summarize(result):
    report = result.get("report", result)
    variants = report["variants"]
    base = variants.get("diffusion:final", {}).get("held_out")
    lines = [
        "# Conditioning experiment results",
        "",
        "| Condition | Steps | All-masked NLL ↓ | Δ vs diffusion final ↓ | Greedy token accuracy | Block ms/token |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, variant in variants.items():
        held = variant["held_out"]
        delta = held["nll"] - base["nll"] if base else None
        accuracy = held.get("generated_accuracy_by_position")
        accuracy = f"{sum(accuracy) / len(accuracy):.2%}" if accuracy else "not measured"
        latency = variant["latency"]["block"]["seconds_per_token"] * 1000
        delta_text = f"{delta:+.4f}" if delta is not None else "—"
        lines.append(
            f"| {name} | {variant['steps']} | {held['nll']:.4f} | {delta_text} | {accuracy} | {latency:.2f} |"
        )
    lines.extend(
        [
            "",
            "Negative deltas favor the condition. These are exploratory point estimates,",
            "not significance tests. All-masked NLL is not iterative sequence likelihood.",
            "Latency does not imply a speedup at matched quality.",
            "",
        ]
    )
    manifest = report.get("manifest")
    if manifest:
        lines.extend(
            [
                f"Model revision: `{manifest['model_revision']}`.",
                f"Source SHA-256: `{manifest['source_sha256']}`.",
                f"GPU: {manifest['gpu']}. Elapsed: {report['elapsed_seconds']:.1f}s.",
                "",
            ]
        )
    for name, variant in variants.items():
        if not variant["completed"]:
            lines.extend([f"**Incomplete training:** {name}.", ""])
        if variant.get("samples"):
            lines.extend([f"## {name}: example continuations", ""])
            for sample in variant["samples"]:
                lines.extend(
                    [f"Prompt: {sample['prompt']}", "", "```text", sample["completion"], "```", ""]
                )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--output")
    args = parser.parse_args()
    rendered = summarize(json.loads(Path(args.result).read_text()))
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered)
