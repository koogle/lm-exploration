"""Compare full-FP32 outputs against retained matching BF16 runs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_samples(path):
    return json.loads(Path(path).read_text())["checkpoints"][-1]["samples"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=Path("../../artifacts/decoding-presets"),
        help="Folder with own-precision-fp32-20261018.json and decoding-presets-20261018.json; outputs are written beside them",
    )
    parser.add_argument(
        "--repetition",
        type=Path,
        default=Path("../../artifacts/repetition"),
        help="Folder with previous-comparison-20261018.json and low-temperature-20261018.json from the repetition study",
    )
    args = parser.parse_args()
    folder = args.folder

    current = read_samples(folder / "own-precision-fp32-20261018.json")
    assert len(current) == 80
    bf16 = read_samples(folder / "decoding-presets-20261018.json")
    earlier = read_samples(args.repetition / "previous-comparison-20261018.json")
    low_temperature = read_samples(args.repetition / "low-temperature-20261018.json")
    comparisons = []
    for sample in current:
        mode = sample["mode"]
        if mode == "topk50_t0.1":
            pool = [s for s in low_temperature if s["mode"].startswith("topk50_t0.1_seed")]
        elif mode == "topk50_t0.8":
            pool = [s for s in earlier if s["mode"].startswith("sampled_08_seed")]
        else:
            pool = [s for s in bf16 if s["mode"] == mode]
        old = next(
            s for s in pool if s["prompt"] == sample["prompt"] and s["seed"] == sample["seed"]
        )
        divergence = next(
            (
                i
                for i, (a, b) in enumerate(zip(old["generation_ids"], sample["generation_ids"]))
                if a != b
            ),
            None,
        )
        comparisons.append(
            {
                "mode": mode,
                "prompt": sample["prompt"],
                "seed": sample["seed"],
                "token_identical": old["generation_ids"] == sample["generation_ids"],
                "first_divergent_position_zero_based": divergence,
                "bf16_repeated_4gram_fraction": old["repeated_4gram_fraction"],
                "fp32_repeated_4gram_fraction": sample["repeated_4gram_fraction"],
                "bf16_completion": old["completion"],
                "fp32_completion": sample["completion"],
            }
        )
    grouped = defaultdict(list)
    for comparison in comparisons:
        grouped[comparison["mode"]].append(comparison)
    summary = {}
    for mode, selected in grouped.items():
        summary[mode] = {
            "samples": len(selected),
            "token_identical_completions": sum(s["token_identical"] for s in selected),
            "bf16_mean_repeated_4gram_fraction": mean(
                s["bf16_repeated_4gram_fraction"] for s in selected
            ),
            "fp32_mean_repeated_4gram_fraction": mean(
                s["fp32_repeated_4gram_fraction"] for s in selected
            ),
        }
    (folder / "precision-comparison.json").write_text(
        json.dumps({"summary": summary, "samples": comparisons}, indent=2) + "\n"
    )
    lines = [
        "# Full-FP32 sensitivity check",
        "",
        "| Setting | Samples | BF16 repeat token4 | FP32 repeat token4 | Token-identical completions |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s['samples']} | {s['bf16_mean_repeated_4gram_fraction']:.2%} | {s['fp32_mean_repeated_4gram_fraction']:.2%} | {s['token_identical_completions']} |"
        )
    lines += [
        "",
        "Same exact model, prompts, BOS prefix, seeds, and completion cap. FP32 disables TF32 for CUDA matmul and cuDNN. Matching BF16 outputs are retained from the original studies. Native token repetition is comparable within this model; this does not establish universal model quality.",
        "",
    ]
    (folder / "precision-results.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
