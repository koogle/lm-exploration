"""Figures from downloaded run outputs.

    python scripts/figures.py --artifacts artifacts --out artifacts/figures

Reads `<artifacts>/<run>/heldout.jsonl` for the training runs and
`<artifacts>/<run>/summary.json` and `predictions.jsonl` for the COCO evaluations, and
writes `training.png`, `training-small-objects.png`, `coco.png`, `recall-by-size.png`,
`precision-recall.png`, `areas.png`
and three `examples*.png` grids.
"""

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from parallel_detection.metrics import best_threshold, ranked_ap, ranked_curve, summarize
from parallel_detection.run import above, fetch_image, read_records

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e4e3dd"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, GREY = "#e87ba4", "#008300", "#6250d6", "#9a998f"
RED = "#e34948"
# One colour per model or run, the same in every chart. Superseded and negative variants
# are grey.
COLOURS = {
    "moondream2": BLUE,
    "moondream31": YELLOW,
    "slots-001": GREY,
    "slots-bins-4": ORANGE,
    "slots-bins-16": AQUA,
    "slots-sizes-none": GREEN,
    "slots-sizes-grouped": GREY,
    "slots-small-16": VIOLET,
    "slots-rank64": MAGENTA,
    "slots-teacher-coco-rank64": RED,
    "before": GREY,
    # The rank-128 head shares the rank-64 head's data, so its colour; a dotted line or
    # hollow marker tells them apart.
    "rank128": RED,
    # Same data plus Gemini 4 Argon labels: the same colour, dash-dot or cross-hatched.
    "argon14k": RED,
    "argon": RED,
}
# Runs whose training labels drop objects under 0.2% of the image.
TRAINING = [
    ("slots-001", "Regressed boxes, 11k images"),
    ("slots-bins-4", "Coordinate bins, 11k images"),
    ("slots-bins-16", "Coordinate bins, 45k images"),
]
# Runs that keep objects down to 0.03% of the image; their held-out set keeps them too.
SMALL_RUNS = [
    ("slots-sizes-none", "48 slots, 23k images"),
    ("slots-sizes-grouped", "48 slots in two size groups, 23k images"),
    ("slots-small-16", "48 slots, 46k images"),
]
# (evaluation run, colour key, label, label offset in points, hollow marker). An
# evaluation that recorded every box's confidence is rescored at the threshold its
# training run chose on held-out images; the colour key names that training run.
L4_POINTS = [
    ("moondream2-001", "moondream2", "Moondream 2 detect", (10, -4), False),
    ("slots-001-coco", "slots-001", "Slots, regressed boxes, 11k", (10, -4), False),
    ("slots-bins-4-coco", "slots-bins-4", "Slots, bins, 11k", (60, -30), False),
    ("slots-bins-16-coco", "slots-bins-16", "Slots, bins, 45k", (60, 10), False),
    ("slots-sizes-none-coco", "slots-sizes-none", "Slots, small kept, 23k", (60, -8), False),
    ("slots-small-16-coco", "slots-small-16", "Slots, small kept, 46k", (60, 32), False),
]
L40S_POINTS = [
    ("moondream31-001", "moondream31", "Moondream 3.1 (Photon)", (10, -4), False),
    ("moondream2-photon-001", "moondream2", "Moondream 2 (Photon)", (10, -4), False),
    ("slots-small-16-coco-l40s", "slots-small-16", "Slots, small kept, 46k", (40, -30), False),
    ("slots-rank64-coco-v2", "slots-rank64", "Slots, rank 64, 46k", (-150, -10), False),
    (
        "slots-teacher-coco-rank64-coco-v2",
        "slots-teacher-coco-rank64",
        "Slots, rank 64, 46k COCO + 18k Gemini-labelled",
        (40, 22),
        False,
    ),
]
# (evaluation run, colour key, label) for the recall-by-size comparison, in an order
# whose neighbouring colours pass the palette check.
SIZE_COMPARISON = [
    ("moondream2-001", "moondream2", "Moondream 2 detect"),
    ("slots-rank64-coco-v2", "slots-rank64", "Slots, rank 64, 46k COCO"),
    ("moondream31-001", "moondream31", "Moondream 3.1"),
    (
        "slots-teacher-coco-rank64-coco-v2",
        "slots-teacher-coco-rank64",
        "Slots, + 18k Gemini-labelled",
    ),
]
EXAMPLES = [
    ("Annotated", None),
    ("Moondream 2", "moondream2-001"),
    ("Moondream 3.1", "moondream31-001"),
    ("Slot head, 46k, small kept", "slots-small-16-coco"),
]


def style(axis, title):
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def curves(axis, artifacts, runs, value):
    placed = []  # end-label heights already used, in data units
    for run, label in runs:
        path = artifacts / run / "heldout.jsonl"
        if not path.exists():
            continue
        points = rows(path)
        minutes = [r["seconds"] / 60 for r in points]
        values = [value(r) for r in points]
        axis.plot(
            minutes,
            values,
            color=COLOURS[run],
            linewidth=2,
            marker="o",
            markersize=4,
            label=label,
        )
        # An end label close to an earlier one is shifted down below it.
        axis.annotate(
            f"{values[-1]:.2f}",
            (minutes[-1], values[-1]),
            textcoords="offset points",
            xytext=(6, -11 * sum(abs(values[-1] - y) < 0.03 for y in placed)),
            va="center",
            fontsize=9,
            color=INK,
        )
        placed.append(values[-1])
    axis.set_xlabel("Training minutes on one H100", fontsize=9, color=MUTED)
    axis.margins(x=0.08)


def training_figure(artifacts, out):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    panels = [
        ("Held-out F1 at IoU 0.5 (best objectness threshold)", lambda r: best(r)[1]),
        ("Held-out IoU of matched boxes", lambda r: r["matched_iou"]),
    ]
    for axis, (title, value) in zip(axes, panels, strict=True):
        style(axis, title)
        curves(axis, artifacts, TRAINING, value)
        axis.set_ylim(0, 0.85)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    figure.suptitle(
        "Training labels without objects under 0.2% of the image",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    figure.tight_layout()
    figure.savefig(out / "training.png", dpi=200, facecolor=SURFACE)
    plt.close(figure)


def best(row):
    """The row's best objectness threshold and its F1."""
    return max(row["f1_iou_0.5"].items(), key=lambda item: item[1])


def small_objects_figure(artifacts, out):
    """Held-out curves of the runs that keep small objects in training and held-out labels."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor=SURFACE)
    panels = [
        ("Held-out F1 at IoU 0.5", lambda r: best(r)[1]),
        (
            "Recall, objects under 1% of the image",
            lambda r: r["recall_by_size_iou_0.5"][best(r)[0]]["small"],
        ),
        ("Recall, larger objects", lambda r: r["recall_by_size_iou_0.5"][best(r)[0]]["large"]),
    ]
    for axis, (title, value) in zip(axes, panels, strict=True):
        style(axis, title)
        curves(axis, artifacts, SMALL_RUNS, value)
        axis.set_ylim(0, 0.9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    figure.suptitle(
        "Training and held-out labels keep objects down to 0.03% of the image",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    figure.tight_layout()
    figure.savefig(out / "training-small-objects.png", dpi=200, facecolor=SURFACE)
    plt.close(figure)


def coco(artifacts, run, training=None):
    """An evaluation's COCO summary, rescored when its predictions recorded every box's
    confidence: at the threshold chosen on the matching tuning-split evaluation (`<run>`
    with its last "-coco-" replaced by "-coco-tune-") if there is one, otherwise at the
    training run's held-out threshold."""
    path = artifacts / run / "summary.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())["datasets"]["coco_val2017"]
    report = artifacts / str(training) / "report.json"
    predictions = artifacts / run / "predictions.jsonl"
    records = read_records(predictions) if predictions.exists() else []
    head, _, suffix = run.rpartition("-coco-")
    tune = artifacts / f"{head}-coco-tune-{suffix}" / "predictions.jsonl"
    if records and "scores" in records[0] and (tune.exists() or report.exists()):
        if tune.exists() and tune.parent.name != run:
            threshold = best_threshold(read_records(tune), "coco_val2017_tune")
        else:
            threshold = json.loads(report.read_text())["threshold"]
        result = summarize(above(records, threshold))["coco_val2017"] | {
            "latency": result["latency"]
        }
    return result


def scatter(axis, artifacts, points, title):
    style(axis, title)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    for run, key, label, offset, hollow in points:
        result = coco(artifacts, run, training=key)
        if result is None:
            continue
        seconds = result["latency"]["image_seconds"]["mean"]
        f1 = result["detection"]["iou_0.5"]["f1"]
        colour = COLOURS[key]
        axis.scatter(
            [seconds],
            [f1],
            s=90,
            color=SURFACE if hollow else colour,
            edgecolor=colour if hollow else SURFACE,
            linewidth=2,
            zorder=3,
        )
        far = abs(offset[0]) > 20 or abs(offset[1]) > 20
        axis.annotate(
            f"{label}\n{f1:.2f} F1, {seconds:.3f} s",
            (seconds, f1),
            textcoords="offset points",
            xytext=offset,
            va="center",
            fontsize=9,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8} if far else None,
        )
    axis.set_xlabel("Seconds per image, all prompted categories", fontsize=9, color=MUTED)
    axis.set_ylim(0.4, 0.75)


def coco_figure(artifacts, out):
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.4), facecolor=SURFACE)
    scatter(axes[0], artifacts, L4_POINTS, "On an L4")
    axes[0].set_xlim(0, 1.4)
    scatter(axes[1], artifacts, L40S_POINTS, "On an L40S")
    axes[1].set_xlim(0, 0.3)
    axes[1].set_ylim(0.55, 0.72)
    axes[0].set_ylabel("F1 at IoU 0.5", fontsize=9, color=MUTED)
    figure.suptitle(
        "COCO val2017, 500 images: accuracy against latency, compared on the same GPU",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    figure.tight_layout()
    figure.savefig(out / "coco.png", dpi=200, facecolor=SURFACE)
    plt.close(figure)


def recall_figure(artifacts, out):
    """Recall by COCO object size, and precision, for the strongest runs and baselines."""
    sizes = ["small", "medium", "large"]
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [3, 1.4]}
    )
    style(left, "Recall at IoU 0.5 by COCO object size")
    style(right, "Precision at IoU 0.5")
    width = 0.8 / len(SIZE_COMPARISON)
    for index, (run, key, label) in enumerate(SIZE_COMPARISON):
        result = coco(artifacts, run, training=key)
        if result is None:
            continue
        detection = result["detection"]["iou_0.5"]
        values = [detection["recall_by_size"][size]["recall"] for size in sizes]
        places = [s + (index - (len(SIZE_COMPARISON) - 1) / 2) * width for s in range(3)]
        bars = left.bar(
            places,
            values,
            width - 0.02,
            color=COLOURS[key],
            edgecolor=SURFACE,
            linewidth=2,
            label=label,
        )
        left.bar_label(bars, labels=[f"{v:.2f}" for v in values], fontsize=8, color=INK, padding=2)
        precision = right.bar(
            [index],
            [detection["precision"]],
            0.7,
            color=COLOURS[key],
            edgecolor=SURFACE,
            linewidth=2,
        )
        right.bar_label(precision, labels=[f"{detection['precision']:.2f}"], fontsize=8, color=INK)
    counts = coco(artifacts, SIZE_COMPARISON[0][0])["detection"]["iou_0.5"]["recall_by_size"]
    left.set_xticks(
        range(3),
        [f"{size}\n({counts[size]['truth_boxes']} boxes)" for size in sizes],
        fontsize=9,
        color=MUTED,
    )
    right.set_xticks(range(len(SIZE_COMPARISON)), ["" for _ in SIZE_COMPARISON])
    for axis in (left, right):
        axis.set_ylim(0, 1)
    left.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left", ncols=2)
    figure.tight_layout()
    figure.savefig(out / "recall-by-size.png", dpi=200, facecolor=SURFACE)
    plt.close(figure)


# Precision-recall comparison: (evaluation run, colour key, label), legend order chosen
# so neighbouring colours pass the palette check. Runs with recorded confidences become
# curves; Moondream gives no confidence, so each model is one point.
PR_SERIES = [
    ("moondream2-001", "moondream2", "Moondream 2 detect"),
    ("slots-rank64-coco-v2", "slots-rank64", "Slots, rank 64, 46k COCO"),
    ("moondream31-001", "moondream31", "Moondream 3.1"),
    (
        "slots-teacher-coco-rank64-coco-v2",
        "slots-teacher-coco-rank64",
        "Slots, + 18k Gemini-labelled",
    ),
    ("slots-teacher-coco-rank128-coco-v2", "rank128", "Same data, rank 128"),
    ("slots-teachers-14k-coco-rank64-coco-v2", "argon14k", "+ 14.7k Argon-labelled"),
    ("slots-teachers-coco-rank64-coco-v2", "argon", "+ 20.8k Argon-labelled"),
    ("slots-teacher-coco-rank64-coco", "before", "Same head before re-scoring (objectness only)"),
]


def pr_figure(artifacts, out):
    """Precision against recall at IoU 0.5 on COCO val, with the area under each curve."""
    figure, axis = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    style(axis, "COCO val2017, 500 images: precision against recall at IoU 0.5")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    for run, key, label in PR_SERIES:
        path = artifacts / run / "predictions.jsonl"
        if not path.exists():
            continue
        records = [r for r in read_records(path) if r["dataset"] == "coco_val2017"]
        colour = COLOURS[key]
        if "scores" in records[0]:
            precision, recall, _ = ranked_curve(records)
            ap = ranked_ap(records)["pooled_ap"]
            axis.plot(
                recall,
                precision,
                color=colour,
                linewidth=2,
                linestyle={
                    "before": "--",
                    "rank128": ":",
                    "argon14k": "-.",
                    "argon": (0, (6, 1.5, 1, 1.5, 1, 1.5)),
                }.get(key, "-"),
                label=f"{label}, area {ap:.3f}",
            )
            if key != "before":
                point = coco(artifacts, run, training=key)["detection"]["iou_0.5"]
                hollow = key == "rank128"
                axis.scatter(
                    [point["recall"]],
                    [point["precision"]],
                    s=60,
                    color=SURFACE if hollow else colour,
                    edgecolor=colour if hollow else SURFACE,
                    linewidth=2,
                    zorder=3,
                )
        else:
            point = summarize(records)["coco_val2017"]["detection"]["iou_0.5"]
            axis.scatter(
                [point["recall"]],
                [point["precision"]],
                s=110,
                marker="D",
                color=colour,
                edgecolor=SURFACE,
                linewidth=2,
                zorder=4,
                label=f"{label} (no confidence: one point)",
            )
    axis.set_xlabel("Recall", fontsize=9, color=MUTED)
    axis.set_ylabel("Precision", fontsize=9, color=MUTED)
    axis.set_xlim(0.3, 0.8)
    axis.set_ylim(0.4, 1.0)
    axis.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    axis.annotate(
        "Dots: each head at the threshold chosen on a separate tuning split.\n"
        "The grey curve stops at confidence 0.3, the lowest it recorded.",
        (0.99, 0.99),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    figure.tight_layout()
    figure.savefig(out / "precision-recall.png", dpi=200, facecolor=SURFACE)
    plt.close(figure)


# Every slot head evaluated on COCO val, in two panels: heads re-scored over all slots,
# and earlier heads ranked by objectness and recorded down to 0.3.
# (evaluation run, colour, line style, label)
ALL_RESCORED = [
    ("slots-rank64-coco-v2", MAGENTA, "-", "Rank 64, 46k COCO"),
    ("slots-rank64-self-coco-v2", MAGENTA, "--", "Same, self-training"),
    ("slots-teacher-coco-rank64-coco-v2", RED, "-", "Rank 64, + 18k Gemini-labelled"),
    ("slots-teacher-coco-rank64-self-coco-v2", RED, "--", "Same, self-training"),
    ("slots-teacher-coco-rank128-coco-v2", RED, ":", "Same data, rank 128"),
    ("slots-teachers-14k-coco-rank64-coco-v2", RED, "-.", "+ 14.7k Argon-labelled, rank 64"),
    (
        "slots-teachers-coco-rank64-coco-v2",
        RED,
        (0, (6, 1.5, 1, 1.5, 1, 1.5)),
        "+ 20.8k Argon-labelled, rank 64",
    ),
]
ALL_OBJECTNESS = [
    ("slots-small-16-coco-sweep-l40s", VIOLET, "-", "Rank 16, 46k COCO"),
    ("slots-iou-coco", AQUA, "-", "Rank 16, IoU objectness target"),
    ("slots-near-coco", GREEN, "-", "Rank 16, near-label negatives"),
    ("slots-rank64-coco", MAGENTA, "-", "Rank 64, 46k COCO"),
    ("slots-combined-coco", ORANGE, "-", "Rank 64, IoU target + near negatives"),
    ("slots-teacher-coco-rank64-coco", RED, "-", "Rank 64, + 18k Gemini-labelled"),
    ("slots-teacher-23k-coco", GREY, "-", "Rank 16, 18k Gemini-labelled only"),
    ("slots-teacher-23k-near-coco", GREY, "--", "Same, near-label negatives"),
]


def baseline_points(axis, artifacts):
    for run, key, label in (
        ("moondream2-001", "moondream2", "Moondream 2 detect"),
        ("moondream31-001", "moondream31", "Moondream 3.1"),
    ):
        path = artifacts / run / "predictions.jsonl"
        if not path.exists():
            continue
        records = [r for r in read_records(path) if r["dataset"] == "coco_val2017"]
        point = summarize(records)["coco_val2017"]["detection"]["iou_0.5"]
        axis.scatter(
            [point["recall"]],
            [point["precision"]],
            s=110,
            marker="D",
            color=COLOURS[key],
            edgecolor=SURFACE,
            linewidth=2,
            zorder=4,
            label=f"{label} (one point)",
        )


def all_pr_figure(artifacts, out):
    """Precision against recall at IoU 0.5 for every slot head evaluated on COCO val."""
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.5), facecolor=SURFACE, sharey=True)
    panels = (
        (axes[0], ALL_RESCORED, "Re-scored heads: every slot, objectness × label × box"),
        (axes[1], ALL_OBJECTNESS, "Earlier heads: objectness only, recorded down to 0.3"),
    )
    for axis, series, title in panels:
        style(axis, title)
        axis.grid(axis="x", color=GRID, linewidth=0.8)
        for run, colour, line, label in series:
            path = artifacts / run / "predictions.jsonl"
            if not path.exists():
                continue
            records = [r for r in read_records(path) if r["dataset"] == "coco_val2017"]
            precision, recall, _ = ranked_curve(records)
            ap = ranked_ap(records)["pooled_ap"]
            axis.plot(
                recall,
                precision,
                color=colour,
                linewidth=2,
                linestyle=line,
                label=f"{label}, area {ap:.3f}",
            )
        baseline_points(axis, artifacts)
        axis.set_xlabel("Recall", fontsize=9, color=MUTED)
        axis.set_xlim(0.2, 0.8)
        axis.set_ylim(0.4, 1.0)
        axis.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    axes[0].set_ylabel("Precision", fontsize=9, color=MUTED)
    axes[1].annotate(
        "Areas here are lower bounds: each curve stops at confidence 0.3.",
        (0.99, 0.99),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    figure.suptitle(
        "COCO val2017, 500 images: precision against recall at IoU 0.5", color=INK, fontsize=11
    )
    figure.tight_layout()
    figure.savefig(out / "precision-recall-all.png", dpi=200, facecolor=SURFACE)
    plt.close(figure)


# Area comparison: evaluations that recorded every box (no confidence floor), each with
# its tuning-split twin. Variants of a head share its colour and differ by texture:
# (evaluation run, colour key, label, hatch or None)
AREA_SERIES = [
    ("slots-rank64-coco-v2", "slots-rank64", "Rank 64, 46k COCO", None),
    ("slots-rank64-self-coco-v2", "slots-rank64", "Same, self-training", "////"),
    (
        "slots-teacher-coco-rank64-coco-v2",
        "slots-teacher-coco-rank64",
        "+ 18k Gemini-labelled",
        None,
    ),
    (
        "slots-teacher-coco-rank64-self-coco-v2",
        "slots-teacher-coco-rank64",
        "Same, self-training",
        "////",
    ),
    (
        "slots-teacher-coco-rank128-coco-v2",
        "slots-teacher-coco-rank64",
        "Same data, rank 128",
        "..",
    ),
    (
        "slots-teachers-14k-coco-rank64-coco-v2",
        "slots-teacher-coco-rank64",
        "+ 14.7k Argon-labelled",
        "xx",
    ),
    (
        "slots-teachers-coco-rank64-coco-v2",
        "slots-teacher-coco-rank64",
        "+ 20.8k Argon-labelled",
        "--",
    ),
]


def area_figure(artifacts, out):
    """Areas under the precision-recall curve for slot heads scored the same way."""
    measures = [
        ("Pooled, IoU 0.5", lambda r: ranked_ap(r)["pooled_ap"]),
        ("Per category, IoU 0.5", lambda r: ranked_ap(r)["category_mean_ap"]),
        ("Per category, IoU 0.75", lambda r: ranked_ap(r, 0.75)["category_mean_ap"]),
        (
            "Per category, IoU 0.5 to 0.95",
            lambda r: float(
                np.mean([ranked_ap(r, t)["category_mean_ap"] for t in np.arange(0.5, 0.96, 0.05)])
            ),
        ),
    ]
    figure, axis = plt.subplots(figsize=(14, 4.8), facecolor=SURFACE)
    style(axis, "COCO val2017, 500 images: area under the precision-recall curve")
    width = 0.8 / len(AREA_SERIES)
    for index, (run, key, label, hatch) in enumerate(AREA_SERIES):
        path = artifacts / run / "predictions.jsonl"
        if not path.exists():
            continue
        records = [r for r in read_records(path) if r["dataset"] == "coco_val2017"]
        values = [measure(records) for _, measure in measures]
        places = [m + (index - (len(AREA_SERIES) - 1) / 2) * width for m in range(len(measures))]
        bars = axis.bar(
            places,
            values,
            width - 0.02,
            color=SURFACE if hatch else COLOURS[key],
            edgecolor=COLOURS[key],
            hatch=hatch,
            linewidth=1.5,
            label=label,
        )
        axis.bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=7, color=INK, padding=2)
    axis.set_xticks(range(len(measures)), [name for name, _ in measures], fontsize=9, color=MUTED)
    axis.set_ylim(0, 0.8)
    axis.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right", ncols=3)
    axis.annotate(
        "Moondream 2 and 3.1 give no confidence scores, so they have no curve or area.",
        (0.0, -0.13),
        xycoords="axes fraction",
        fontsize=8,
        color=MUTED,
    )
    figure.tight_layout()
    figure.savefig(out / "areas.png", dpi=200, facecolor=SURFACE)
    plt.close(figure)


def by_image(path):
    grouped = defaultdict(list)
    for record in rows(path):
        if record["dataset"] == "coco_val2017":
            grouped[record["image_id"]].append(record)
    return grouped


def examples_figure(artifacts, out, name, objects, skip=0, count=4):
    """Side-by-side boxes on `count` images whose annotated object count is in `objects`."""
    runs = {run: by_image(artifacts / run / "predictions.jsonl") for _, run in EXAMPLES if run}
    reference = runs["moondream2-001"]
    low, high = objects
    chosen = [
        image_id
        for image_id, records in sorted(reference.items())
        if low <= sum(len(r["boxes"]) for r in records) <= high and len(records) >= 2
    ][skip : skip + count]
    figure, axes = plt.subplots(count, len(EXAMPLES), figsize=(13, 3.1 * count), facecolor=SURFACE)
    for row, image_id in enumerate(chosen):
        image = fetch_image(
            f"http://images.cocodataset.org/val2017/{image_id:012d}.jpg", out / "images"
        )
        for axis, (title, run) in zip(axes[row], EXAMPLES, strict=True):
            records = runs[run][image_id] if run else reference[image_id]
            boxes = [b for r in records for b in (r["predicted"] if run else r["boxes"])]
            axis.imshow(image)
            for x0, y0, x1, y1 in boxes:
                axis.add_patch(
                    Rectangle(
                        (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=SURFACE, linewidth=3.5
                    )
                )
                axis.add_patch(
                    Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor=BLUE if run else INK,
                        linewidth=1.8,
                    )
                )
            caption = f"{len(boxes)} boxes"
            if run:
                seconds = records[0]["shared_seconds"] + sum(r["seconds"] for r in records)
                caption += f", {seconds:.2f} s"
            else:
                phrases = ", ".join(r["phrase"] for r in records)
                caption += ": " + textwrap.shorten(phrases, 44, placeholder=" …")
            axis.set_title(
                f"{title}\n{caption}" if row == 0 else caption, fontsize=9, color=INK, loc="left"
            )
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(out / name, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    training_figure(args.artifacts, args.out)
    small_objects_figure(args.artifacts, args.out)
    coco_figure(args.artifacts, args.out)
    recall_figure(args.artifacts, args.out)
    pr_figure(args.artifacts, args.out)
    all_pr_figure(args.artifacts, args.out)
    area_figure(args.artifacts, args.out)
    examples_figure(args.artifacts, args.out, "examples.png", objects=(4, 9))
    examples_figure(args.artifacts, args.out, "examples-2.png", objects=(4, 9), skip=4)
    examples_figure(args.artifacts, args.out, "examples-crowded.png", objects=(12, 25))


if __name__ == "__main__":
    main()
