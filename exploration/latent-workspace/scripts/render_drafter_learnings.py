"""Export the short research write-up and three evidence charts as one page."""

import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(".")
text = (
    root / "docs/implementing-a-diffusion-style-drafter-from-scratch-and-learnings.md"
).read_text()
paragraphs = text.split("\n\n")[1:6]
plain = [p.replace("**", "").replace("→", " → ") for p in paragraphs]
left = "\n\n".join(textwrap.fill(p, width=84) for p in plain[:3])
right = "\n\n".join(textwrap.fill(p, width=84) for p in plain[3:])
fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
fig.text(
    0.035, 0.955, "Implementing and training a diffusion-style drafter", fontsize=18, weight="bold"
)
fig.text(0.035, 0.921, "From scratch: learnings", fontsize=14, color="#274963")
fig.text(0.035, 0.875, left, fontsize=9.1, va="top", linespacing=1.18)
fig.text(0.525, 0.875, right, fontsize=9.1, va="top", linespacing=1.18)
axes = [fig.add_axes([x, 0.12, 0.28, 0.26]) for x in (0.055, 0.38, 0.705)]
a = axes[0]
a.bar(
    range(4),
    [0.384436, 0.484354, 0.519650, 0.530051],
    color=["#78909c", "#4c78a8", "#4c78a8", "#4c78a8"],
)
a.set(
    xticks=range(4),
    xticklabels=["Original", "+ Q/K norm", "5 layers", "5 depths"],
    ylim=(0, 0.65),
    title="Matched early architecture checks",
    ylabel="Validation NEW / 7",
)
a.tick_params(axis="x", rotation=20)
for i, v in enumerate([0.384, 0.484, 0.520, 0.530]):
    a.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
r = json.loads((root / "results/large-curriculum-final.json").read_text())["reports"]["256"]
c = [p for p in r["curve"] if p["step"] > r["curriculum_transition"]["step"]]
a = axes[1]
a.plot([p["step"] / 1000 for p in c], [p["validation"]["nll"] for p in c], label="Validation")
a.plot(
    [p["step"] / 1000 for p in c],
    [p["training_diagnostic"]["nll"] for p in c],
    linestyle="--",
    label="12 training examples",
)
a.set(title="Long fit: late overfitting", xlabel="Updates (thousands)", ylabel="NLL")
a.legend(fontsize=7)
r = json.loads((root / "results/qualification-long-196k-01.json").read_text())
a = axes[2]
keys = ["prose", "code", "translation"]
x = range(3)
a.bar(
    [i - 0.18 for i in x],
    [
        r["frozen"]["selected_validation"]["by_domain"][d]["mean_accepted_beyond_anchor"]
        for d in keys
    ],
    0.36,
    label="Validation",
)
a.bar(
    [i + 0.18 for i in x],
    [r["test_metrics"]["by_domain"][d]["mean_accepted_beyond_anchor"] for d in keys],
    0.36,
    label="Sealed test",
)
a.set(
    xticks=list(x),
    xticklabels=["Prose", "Code", "Translation"],
    ylim=(0, 7),
    title="Frozen held-out result",
    ylabel="Correct NEW / 7",
)
a.legend(fontsize=7)
for a in axes:
    a.tick_params(labelsize=8)
    a.title.set_fontsize(10)
    a.grid(axis="y", alpha=0.15)
fig.text(
    0.035,
    0.045,
    "One seed; different experiments have different exposure budgets. Training diagnostic is tiny. Matching Qwen is not a human-quality score.",
    fontsize=8,
)
fig.text(
    0.035,
    0.024,
    "Sources: DFlash arXiv:2602.06036; Anthropic workspace/Jacobian lens; full reports and protocol in the repository.",
    fontsize=8,
)
fig.savefig(root / "results/drafter-learnings.png", dpi=160)
fig.savefig(root / "results/drafter-learnings.pdf")
print("Body words:", len(re.findall(r"\S+", " ".join(paragraphs))))
