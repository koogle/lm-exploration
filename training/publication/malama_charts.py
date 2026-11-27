"""Charts for the Malama model card, from its results file.

Run from training/:
    uv run --extra research python -m publication.malama_charts \
        --results artifacts/malama/results.json --output artifacts/malama/charts

`--results` is the release's `results.json` (kept beside the card on Hugging Face): one
section per protocol, each a list of models with `name`, `params` (billions), `ours` and
that protocol's scores. Writes one PNG per chart; Malama is drawn in warm terracotta and
reference models in cool slate, with each point or bar labelled by name.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OURS = "#d95f2b"
OURS_SOFT = "#f2b291"
OTHER = "#94a3b8"
INK = "#0f172a"
LABEL = "#334155"
MUTED = "#64748b"
GRID = "#e2e8f0"
# Lines of the RL-step chart, one per reasoning SFT run, progressing to the final run.
RUNS = {"v3": "SFT v3", "v4": "SFT v4", "v5": "SFT v5", "mix": "SFT v5, mixed RL"}
RUN_COLORS = {"v3": "#94a3b8", "v4": "#5b8fb9", "v5": "#1d4e89", "mix": OURS}
GROUP_SHADES = ("#cbd5e1", "#64748b", "#334155", "#1e293b")


def axes(width=8.0, height=4.6):
    figure, axis = plt.subplots(figsize=(width, height), dpi=160, facecolor="white")
    axis.set_facecolor("white")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    return figure, axis


def save(figure, axis, title, path):
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)
    figure.tight_layout()
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def by_size(models, value, ylabel, title, path, offsets=None):
    """Score against parameters (log scale), every model labelled."""
    figure, axis = axes()
    for model in models:
        color = OURS if model["ours"] else OTHER
        axis.scatter(model["params"], value(model), s=84 if model["ours"] else 46, color=color,
                     edgecolors="white", linewidths=1.5, zorder=4 if model["ours"] else 3)  # fmt: skip
        dx, dy, align = (offsets or {}).get(model["name"], (7, 0, "left"))
        axis.annotate(model["name"], (model["params"], value(model)), xytext=(dx, dy),
                      textcoords="offset points", ha=align, va="center", fontsize=8.5,
                      color=OURS if model["ours"] else LABEL,
                      fontweight="bold" if model["ours"] else "normal")  # fmt: skip
    axis.set_xscale("log")
    ticks = [0.25, 0.5, 1, 2, 4]
    axis.set_xticks(ticks, [f"{tick:g}B" for tick in ticks])
    axis.minorticks_off()
    axis.set_xlim(0.2, 8)
    axis.set_xlabel("Parameters (log scale)", fontsize=9, color=MUTED)
    axis.set_ylabel(ylabel, fontsize=9, color=MUTED)
    save(figure, axis, title, path)


def by_step(models, branches, path):
    """LiveCodeBench pass@1 against total RL steps, one line per run; a run in `branches`
    starts from the checkpoint of the run it continues."""
    figure, axis = axes()
    for run, label in RUNS.items():
        points = sorted((m["step"], m["p1"]["all"]) for m in models if m.get("run") == run)
        if run in branches:
            parent = branches[run]
            points = [
                (m["step"], m["p1"]["all"])
                for m in models
                if m.get("run") == parent["run"] and m["step"] == parent["step"]
            ] + points
        steps, scores = zip(*points, strict=True)
        is_final = run == "mix"
        axis.plot(steps, scores, color=RUN_COLORS[run], linewidth=2.5 if is_final else 2.0,
                  marker="o", markersize=6.5 if is_final else 5.5,
                  markeredgecolor="white", markeredgewidth=1.5, label=label,
                  zorder=4 if is_final else 3)  # fmt: skip
        axis.annotate(f"{scores[-1]:.1f}", (steps[-1], scores[-1]), xytext=(7, 0),
                      textcoords="offset points", va="center", fontsize=8.5,
                      color=OURS if is_final else LABEL,
                      fontweight="bold" if is_final else "normal")  # fmt: skip
    axis.set_xlabel("RL steps after the run's reasoning SFT", fontsize=9, color=MUTED)
    axis.set_ylabel("pass@1 (%)", fontsize=9, color=MUTED)
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    save(figure, axis, "LiveCodeBench v6 with thinking, by RL step", path)


def grouped(models, keys, labels, title, path):
    """One group of bars per benchmark, one bar per model, each labelled with its value."""
    figure, axis = axes(9.0, 4.8)
    width = 0.8 / len(models)
    shades = iter(GROUP_SHADES)
    for index, model in enumerate(models):
        color = (
            (OURS if "v0.0.1" in model["name"] else OURS_SOFT) if model["ours"] else next(shades)
        )
        places = [group + (index - (len(models) - 1) / 2) * width for group in range(len(keys))]
        bars = axis.bar(places, [model[key] for key in keys], width * 0.92, color=color,
                        label=model["name"])  # fmt: skip
        axis.bar_label(bars, fmt="%.1f", fontsize=7.5, color=LABEL, padding=2)
    axis.set_xticks(range(len(keys)), labels)
    axis.set_ylabel("pass@1 (%)", fontsize=9, color=MUTED)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left", ncols=2)
    axis.set_ylim(0, 105)
    save(figure, axis, title, path)


def ranked(models, key, title, path):
    """Horizontal bars ranked by `key`, best first."""
    models = sorted(models, key=lambda model: model[key])
    figure, axis = axes(7.5, 0.42 * len(models) + 1.2)
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    bars = axis.barh([m["name"] for m in models], [m[key] for m in models], height=0.62,
                     color=[OURS if m["ours"] else OTHER for m in models])  # fmt: skip
    axis.bar_label(bars, fmt="%.1f", fontsize=8.5, color=LABEL, padding=4)
    axis.set_xlim(0, max(m[key] for m in models) * 1.12)
    axis.set_xlabel("pass@1 (%)", fontsize=9, color=MUTED)
    axis.tick_params(axis="y", labelsize=9, colors=INK)
    save(figure, axis, title, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    thinking = results["livecodebench_thinking"]["models"]
    best = max((m for m in thinking if m["ours"]), key=lambda m: m["p1"]["all"])
    shown = [m for m in thinking if not m["ours"]] + [{**best, "name": results["release"]}]
    by_size(shown, lambda m: m["p1"]["all"], "pass@1 (%)",
            "LiveCodeBench v6 with thinking, by model size", output / "livecodebench-by-size.png",
            {"MobileLLM-R1.5-950M": (-7, 0, "right"), results["release"]: (-9, 0, "right")})  # fmt: skip
    by_step(
        thinking,
        results["livecodebench_thinking"].get("branches", {}),
        output / "livecodebench-by-rl-step.png",
    )
    grouped(results["chat_function_tasks"]["models"], ["livecodebench", "humaneval", "mbpp"],
            ["LiveCodeBench v6", "HumanEval", "MBPP"],
            "Chat with thinking: contest and function problems",
            output / "chat-benchmarks.png")  # fmt: skip
    by_size(results["perplexity"]["models"], lambda m: m["arxiv_bits_per_byte"],
            "Bits per byte (lower is better)", "Base models on arXiv abstracts of 2026-11-24",
            output / "perplexity-by-size.png",
            {"Malama Base": (-4, 11, "center"), "SmolLM2-360M": (-7, 0, "right"), "Qwen3-0.6B-Base": (-7, -2, "right"),
             "Qwen3.5-0.8B-Base": (-7, 5, "right"), "OLMo-2-1B": (7, -5, "left"),
             "MobileLLM-R1-950M-base": (7, 2, "left")})  # fmt: skip
    for key, label in (("humaneval", "HumanEval, zero-shot"), ("mbpp", "MBPP, 3-shot")):
        ranked(results["plain_completion"]["models"], key, f"Plain completion: {label}",
               output / f"completion-{key}.png")  # fmt: skip
    ranked(results["infilling"]["models"], "single", "HumanEval-Infilling, single line",
           output / "infilling-single-line.png")  # fmt: skip


if __name__ == "__main__":
    main()
