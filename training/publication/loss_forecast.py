"""Fit a loss forecast to a run's early training loss and chart the run against it.

Run from training/:
    uv run --extra research python -m publication.loss_forecast fit \
        --metrics lineage-metrics.jsonl --until 26.4 --output forecast.json
    uv run --extra research python -m publication.loss_forecast plot \
        --metrics lineage-metrics.jsonl --validation lineage-validation.jsonl \
        --forecast forecast.json --decay-start 85 --output training-loss-forecast.png

`--metrics` and `--validation` are a run's metrics and held-out records (one JSON object
per line with `tokens` and `loss`, or `tokens` and `heldout`). The forecast is
`loss = c + a * tokens_b ** -b`, fitted by least squares to the mean loss of 0.5B-token
bins, over three windows that end at `--until` billion tokens; the middle window is the
central forecast and the three give its range.
"""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

WINDOW_STARTS = (2.0, 5.0, 10.0)


def records(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.startswith("{")
    ]


def fit(tokens_b, loss, start, end, width=0.5):
    """Best (c, a, b) of c + a * t^-b over bin means in [start, end), b on a 0.005 grid."""
    edges = np.arange(start, end + 1e-9, width)
    centers = (edges[:-1] + edges[1:]) / 2
    means = np.array(
        [loss[(tokens_b >= lo) & (tokens_b < hi)].mean() for lo, hi in pairwise(edges)]
    )
    best = None
    for b in np.linspace(0.05, 2.0, 391):
        design = np.stack([np.ones_like(centers), centers**-b], 1)
        (c, a), *_ = np.linalg.lstsq(design, means, rcond=None)
        error = float(((design @ (c, a) - means) ** 2).mean())
        if best is None or error < best["mse"]:
            best = {
                "window": [start, end],
                "c": float(c),
                "a": float(a),
                "b": float(b),
                "mse": error,
            }
    best["rmse"] = best.pop("mse") ** 0.5
    return best


def predict(curve, tokens_b):
    return curve["c"] + curve["a"] * np.asarray(tokens_b, dtype=float) ** -curve["b"]


def forecast(rows, until):
    tokens_b = np.array([row["tokens"] for row in rows]) / 1e9
    loss = np.array([row["loss"] for row in rows])
    keep = tokens_b < until
    end = float(np.floor(until))
    fits = [fit(tokens_b[keep], loss[keep], start, end) for start in WINDOW_STARTS]
    return {
        "fitted_at_tokens_b": until,
        "form": "loss = c + a * tokens_b ** -b",
        "fits": fits,
        "central": fits[1],
    }


def plot(rows, heldout, curves, decay_start, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from publication.malama_charts import GRID, INK, MUTED, OTHER, OURS

    tokens_b = np.array([row["tokens"] for row in rows]) / 1e9
    loss = np.array([row["loss"] for row in rows])
    window = 200
    smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5), dpi=160, facecolor="white")
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(GRID)
        axis.tick_params(colors=MUTED, labelsize=9, length=0)
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel("Training tokens (billions)", fontsize=9, color=MUTED)
    left.plot(
        tokens_b[window - 1 :],
        smooth,
        color=OURS,
        linewidth=1.8,
        label=f"Training loss, {window}-step mean",
    )
    span = np.linspace(2, decay_start, 300)
    predictions = np.array([predict(curve, span) for curve in curves["fits"]])
    left.fill_between(
        span, predictions.min(0), predictions.max(0), color=OTHER, alpha=0.3, linewidth=0
    )
    left.plot(span, predict(curves["central"], span), color=INK, linewidth=1.2, linestyle=(0, (4, 3)),
              label=f"Forecast fitted at {curves['fitted_at_tokens_b']:.0f}B tokens")  # fmt: skip
    for x, text in (
        (curves["fitted_at_tokens_b"], "forecast frozen"),
        (decay_start, "learning-rate decay"),
    ):
        left.axvline(x, color=MUTED, linewidth=0.8, linestyle=":")
        left.annotate(
            text, (x, 2.55), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=MUTED
        )
    left.set_ylim(1.7, 2.6)
    left.set_ylabel("Next-token loss (nats per token)", fontsize=9, color=MUTED)
    left.legend(frameon=False, fontsize=8.5, loc="lower left")
    left.set_title(
        "Training loss against the forecast", loc="left", fontsize=12, fontweight="bold", color=INK
    )
    points = [row for row in heldout if row["tokens"] >= 2e9]
    held_tokens = np.array([row["tokens"] for row in points]) / 1e9
    ends = []
    for source in sorted(points[-1]["heldout"]):
        values = [row["heldout"][source]["bits_per_byte"] for row in points]
        ends.append((values[-1], source))
        right.plot(
            held_tokens, values, linewidth=1.3, color=OURS if source == "climbmix" else OTHER
        )
    placed = []
    for value, source in sorted(ends):
        y = value if not placed else max(value, placed[-1] + 0.025)
        placed.append(y)
        right.annotate(f"{source} {value:.2f}", (held_tokens[-1], value), xytext=(held_tokens[-1] * 1.03, y),
                       textcoords="data", fontsize=8, va="center", color=INK,
                       arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.8})  # fmt: skip
    right.set_xlim(0, held_tokens[-1] * 1.3)
    right.set_ylabel("Held-out bits per byte", fontsize=9, color=MUTED)
    right.set_title(
        "Held-out loss by source", loc="left", fontsize=12, fontweight="bold", color=INK
    )
    figure.tight_layout()
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit")
    fitting.add_argument("--metrics", required=True)
    fitting.add_argument(
        "--until", type=float, required=True, help="Fit to tokens before this, in billions"
    )
    fitting.add_argument("--output", required=True)
    charting = commands.add_parser("plot")
    charting.add_argument("--metrics", required=True)
    charting.add_argument("--validation", required=True)
    charting.add_argument("--forecast", required=True)
    charting.add_argument("--decay-start", type=float, required=True, help="Billions of tokens")
    charting.add_argument("--output", required=True)
    arguments = parser.parse_args()
    rows = records(arguments.metrics)
    if arguments.command == "fit":
        Path(arguments.output).write_text(
            json.dumps(forecast(rows, arguments.until), indent=1) + "\n"
        )
    else:
        curves = json.loads(Path(arguments.forecast).read_text())
        plot(rows, records(arguments.validation), curves, arguments.decay_start, arguments.output)


if __name__ == "__main__":
    main()
