"""Summarize explicit longer-study artifacts, keeping incomplete runs visible."""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "baseline": "Original conv-4",
    "baseline_bf16": "Conv-4, BF16 recurrence inputs",
    "none": "No convolution",
    "rotating_triton": "Rotating memory (Triton)",
    "rope": "RoPE without convolution",
    "middle_rope": "Middle KDA: RoPE replaces conv",
}


def load_launch(folder):
    launch = json.loads((folder / "launch.json").read_text())
    if (folder / "precision-launch.json").exists():
        launch["variants"] = [*launch["variants"], "baseline_bf16"]
    return launch


def analyze(folder):
    launch = load_launch(folder)
    benchmark_path = folder / (
        "benchmark-precision.json"
        if (folder / "precision-launch.json").exists()
        else "benchmark.json"
    )
    benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
    runs = {}
    statuses = []
    for variant in launch["variants"]:
        for seed in launch["seeds"]:
            path = folder / f"{variant}-{seed}.json"
            result = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
            statuses.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "status": result["status"],
                    "steps_completed": result.get("steps_completed", 0),
                }
            )
            if result.get("curve"):
                runs[variant, seed] = result
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for variant in launch["variants"]:
        selected = [runs.get((variant, seed)) for seed in launch["seeds"]]
        if any(run is None for run in selected):
            continue
        curves = [{point["step"]: point for point in run["curve"]} for run in selected]
        common = sorted(set.intersection(*(set(curve) for curve in curves)))
        values = np.array([[curve[step]["validation_ce"] for step in common] for curve in curves])
        tokens = [curves[0][step]["token_positions"] / 1e6 for step in common]
        (line,) = axes[0].plot(tokens, values.mean(0), label=LABELS[variant])
        axes[0].fill_between(
            tokens, values.min(0), values.max(0), color=line.get_color(), alpha=0.12
        )
        complete = all(run["status"] == "complete" for run in selected)
        endpoint = common[-1]
        row = {
            "variant": variant,
            "all_seeds_complete": complete,
            "step": endpoint,
            "mean_validation_ce": float(values[:, -1].mean()),
            "seed_validation_ce": values[:, -1].tolist(),
            "parameters": [run["parameters"] for run in selected],
            "tokens_per_second": [curve[endpoint]["tokens_per_second"] for curve in curves],
            "peak_memory_bytes": [run["max_memory_bytes"] for run in selected],
        }
        if benchmark is not None:
            seconds = benchmark["seconds_per_step"]
            row["same_gpu_ms_per_step"] = float(np.mean(seconds[variant]) * 1000)
            row["same_gpu_speedup_vs_baseline"] = float(
                np.mean(seconds["baseline"]) / np.mean(seconds[variant])
            )
            if "baseline_bf16" in seconds:
                row["same_gpu_speedup_vs_bf16_baseline"] = float(
                    np.mean(seconds["baseline_bf16"]) / np.mean(seconds[variant])
                )
        baselines = [runs.get(("baseline", seed)) for seed in launch["seeds"]]
        if all(baseline is not None for baseline in baselines):
            base_curves = [{p["step"]: p for p in baseline["curve"]} for baseline in baselines]
            paired_steps = sorted(set(common).intersection(*(set(c) for c in base_curves)))
            if paired_steps:
                deltas = np.array(
                    [
                        [
                            curve[step]["validation_ce"] - base[step]["validation_ce"]
                            for step in paired_steps
                        ]
                        for curve, base in zip(curves, base_curves)
                    ]
                )
                x = [curves[0][step]["token_positions"] / 1e6 for step in paired_steps]
                axes[1].plot(x, deltas.mean(0), color=line.get_color(), label=LABELS[variant])
                axes[1].fill_between(
                    x, deltas.min(0), deltas.max(0), color=line.get_color(), alpha=0.12
                )
                row["paired_comparison_step"] = paired_steps[-1]
                row["paired_ce_delta_vs_baseline"] = deltas[:, -1].tolist()
        rows.append(row)
        matched = [runs.get(("baseline_bf16", seed)) for seed in launch["seeds"]]
        if all(control is not None for control in matched):
            reference = [{p["step"]: p for p in control["curve"]} for control in matched]
            shared = sorted(set(common).intersection(*(set(c) for c in reference)))
            if shared:
                row["paired_bf16_comparison_step"] = shared[-1]
                row["paired_ce_delta_vs_bf16_baseline"] = [
                    curve[shared[-1]]["validation_ce"] - control[shared[-1]]["validation_ce"]
                    for curve, control in zip(curves, reference)
                ]
    for ax in axes:
        ax.set_xlabel("Training token positions (millions)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out cross-entropy (nats/token)")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Paired CE difference from baseline (lower is better)")
    axes[1].axhline(0, color="black", linewidth=0.6)
    fig.suptitle("100M-class KDA hybrid: two seeds; bands show seed range")
    fig.tight_layout()
    fig.savefig(folder / "validation-loss.png", dpi=180)
    result = {"run": launch["run"], "statuses": statuses, "comparisons": rows}
    (folder / "summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    launch = load_launch(args.results)
    stop = min(time.time() + args.wait_seconds, launch["deadline"] + 120)
    expected = [
        args.results / f"{variant}-{seed}.json"
        for variant in launch["variants"]
        for seed in launch["seeds"]
    ]
    while time.time() < stop and not all(path.exists() for path in expected):
        time.sleep(min(10, max(0, stop - time.time())))
    analyze(args.results)
    if args.publish:
        import truststore

        truststore.inject_into_ssl()
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            repo_id="jfrick/lm-exploration-results",
            repo_type="dataset",
            folder_path=str(args.results),
            path_in_repo="training/results/" + launch["run"],
            commit_message="Archive paired 100M KDA study comparisons",
        )
