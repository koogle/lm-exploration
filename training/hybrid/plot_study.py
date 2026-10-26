"""Plot measured validation curves from an explicit pilot artifact directory."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    variants = ["baseline", "none", "rope", "middle_rope", "middle_rope_conv", "rotating"]
    labels = {
        "baseline": "Original convolution",
        "none": "No convolution",
        "rope": "No convolution + RoPE",
        "middle_rope": "Middle: replace conv with RoPE",
        "middle_rope_conv": "Middle: add RoPE, keep conv",
        "rotating": "Rotating memory",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in variants:
        path = args.results / f"{variant}.json"
        if not path.exists():
            continue
        runs = json.loads(path.read_text())
        steps = [p["step"] for p in runs[0]["curve"]]
        values = np.array([[p["validation_ce"] for p in run["curve"]] for run in runs])
        (line,) = ax.plot(steps, values.mean(0), label=labels[variant])
        ax.fill_between(steps, values.min(0), values.max(0), alpha=0.12, color=line.get_color())
    ax.set(
        xlabel="Optimizer updates (4,096 token positions/update)",
        ylabel="Held-out cross-entropy (nats/token)",
        title="PR77 small hybrid: two seeds; bands show seed range",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.results / "validation-loss.png", dpi=180)


if __name__ == "__main__":
    main()
