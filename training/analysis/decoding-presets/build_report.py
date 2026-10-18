"""Build inspectable output tables; no model execution or subjective auto-grading."""

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


def repeat_fraction(items, n):
    grams = [tuple(items[i : i + n]) for i in range(max(0, len(items) - n + 1))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def main():
    report = json.loads(Path("decoding-presets-20261018.json").read_text())
    samples = report["checkpoints"][0]["samples"]
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["mode"]].append(sample)
    summary = {}
    for mode, selected in grouped.items():
        prose = [s for s in selected if not (s["prompt"].startswith("def ") or s["prompt"].startswith("import json"))]
        code = [s for s in selected if s not in prose]
        summary[mode] = {
            "samples": len(selected),
            "temperature": selected[0]["temperature"],
            "top_p": selected[0]["top_p"],
            "repetition_penalty": selected[0]["repetition_penalty"],
            "repeat_token4": mean(s["repeated_4gram_fraction"] for s in selected),
            "repeat_token4_prose": mean(s["repeated_4gram_fraction"] for s in prose),
            "repeat_token4_code": mean(s["repeated_4gram_fraction"] for s in code),
            "repeat_char16_first256": mean(repeat_fraction(s["completion"][:256], 16) for s in selected),
            "repeat_normalized_char16_first256": mean(repeat_fraction(re.sub(r"\s+", " ", s["completion"].lower()).strip()[:256], 16) for s in selected),
            "mean_generated_tokens": mean(s["generated_tokens"] for s in selected),
            "eos_count": sum(s["ended_with_eos"] for s in selected),
            "code_completions": len(code),
        }
    Path("summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    previous = json.loads(Path("../repetition/previous-comparison-20261018.json").read_text())
    old_samples = previous["checkpoints"][-1]["samples"]
    comparisons = []
    for old_mode, new_mode in (("greedy", "greedy"), ("sampled_10_topp95", "baseline_t1_p95")):
        for old in (s for s in old_samples if s["mode"] == old_mode):
            new = next(s for s in samples if s["mode"] == new_mode and s["seed"] == old["seed"] and s["prompt"] == old["prompt"])
            comparisons.append({"mode": new_mode, "prompt": old["prompt"], "seed": old["seed"], "token_identical": old["generation_ids"] == new["generation_ids"]})
    assert len(comparisons) == 20
    Path("baseline-reproducibility.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    lines = ["# All decoding study completions", "", "Verbatim outputs; no filtering or cherry-picking.", ""]
    for i, sample in enumerate(samples):
        lines += [f"## Sample {i + 1}: {sample['mode']}, seed {sample['seed']}", "", "Prompt:", "", "````text", sample["prompt"], "````", "", "Completion:", "", "````text", sample["completion"], "````", ""]
    Path("all-completions.md").write_text("\n".join(lines))
    rows = ["| Preset | T | Top-p | Penalty | Repeat token 4-grams | Prose | Code |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary.items():
        rows.append(f"| {name} | {s['temperature']:.2f} | {s['top_p']:.2f} | {s['repetition_penalty']:.2f} | {s['repeat_token4']:.1%} | {s['repeat_token4_prose']:.1%} | {s['repeat_token4_code']:.1%} |")
    Path("results-table.md").write_text("\n".join(rows) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
