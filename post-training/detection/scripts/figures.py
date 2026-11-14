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
# Side-by-side examples per dataset: the panels, (title, evaluation run or None for the
# annotation), and where the dataset's images are.
EXAMPLES = {
    "coco_val2017": [
        ("Annotated", None),
        ("Moondream 2", "moondream2-001"),
        ("Moondream 3.1", "moondream31-001"),
        ("Slot head, tiled", "slots-teacher-coco-rank128-coco-final"),
    ],
    "voc2007_test": [
        ("Annotated", None),
        ("Moondream 2", "moondream2-photon-voc"),
        ("Moondream 3.1", "moondream31-voc"),
        ("Slot head, tiled", "slots-teacher-coco-rank128-voc-final"),
    ],
    "mot17_train": [
        ("Annotated", None),
        ("Moondream 3.1", "moondream31-mot"),
        ("Slot head, fast", "slots-teacher-coco-rank128-mot-fast"),
        ("Slot head, tiled", "slots-teacher-coco-rank128-mot-tiled"),
    ],
}


def example_image(dataset, image_id, images):
    """The image of an example: COCO's from its server, the others from the folder the
    evaluation harness wrote them to (`run.build_samples`)."""
    from PIL import Image

    if dataset == "coco_val2017":
        return fetch_image(f"http://images.cocodataset.org/val2017/{image_id:012d}.jpg", images)
    folder = {"voc2007_test": "voc2007", "mot17_train": "mot17"}[dataset]
    return Image.open(images / folder / f"{image_id}.jpg").convert("RGB")


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


BASELINES = {
    "coco_val2017": ("moondream2-001", "moondream31-001"),
    "voc2007_test": ("moondream2-photon-voc", "moondream31-voc"),
}


def baseline_points(axis, artifacts, dataset="coco_val2017"):
    moondream2, moondream31 = BASELINES[dataset]
    for run, key, label in (
        (moondream2, "moondream2", "Moondream 2 detect"),
        (moondream31, "moondream31", "Moondream 3.1"),
    ):
        path = artifacts / run / "predictions.jsonl"
        if not path.exists():
            continue
        records = [r for r in read_records(path) if r["dataset"] == dataset]
        point = summarize(records)[dataset]["detection"]["iou_0.5"]
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


# Pascal VOC 2007 test: slot heads at the threshold tuned on COCO (none is tuned on VOC).
# (evaluation run, colour, line style, label, COCO-tuned threshold)
VOC_SERIES = [
    ("slots-rank64-voc", MAGENTA, "-", "Slots, rank 64, 46k COCO", 0.0133),
    ("slots-teacher-coco-rank64-voc", RED, "-", "Slots, + 18k Gemini-labelled", 0.0133),
    ("slots-teacher-coco-rank128-voc", RED, ":", "Same data, rank 128", 0.0133),
    (
        "slots-teachers-coco-rank64-voc",
        RED,
        (0, (6, 1.5, 1, 1.5, 1, 1.5)),
        "+ 20.8k Argon-labelled, rank 64",
        0.0152,
    ),
]


def voc_figure(artifacts, out):
    """Precision against recall at IoU 0.5 on Pascal VOC 2007 test, a dataset no model
    here was trained or tuned on."""
    figure, axis = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    style(axis, "Pascal VOC 2007 test, 1,000 images: precision against recall at IoU 0.5")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    for run, colour, line, label, threshold in VOC_SERIES:
        path = artifacts / run / "predictions.jsonl"
        if not path.exists():
            continue
        records = list(read_records(path))
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
        point = summarize(above(records, threshold))["voc2007_test"]["detection"]["iou_0.5"]
        axis.scatter([point["recall"]], [point["precision"]], s=50, color=colour, zorder=3)
    baseline_points(axis, artifacts, "voc2007_test")
    axis.set_xlabel("Recall", fontsize=9, color=MUTED)
    axis.set_ylabel("Precision", fontsize=9, color=MUTED)
    axis.set_xlim(0.6, 1.0)
    axis.set_ylim(0.5, 1.0)
    axis.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    axis.annotate(
        "Dots: each head at the threshold chosen on COCO.",
        (0.99, 0.99),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    figure.tight_layout()
    figure.savefig(out / "voc-precision-recall.png", dpi=200, facecolor=SURFACE)
    plt.close(figure)


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


# Heads compared under the current selection (fusion at IoU 0.5, box confidence^0.25),
# fast (full image) and tiled, from `scripts/study_selection.py` outputs
# (`<run>-selection.json`). (label, run)
SELECTION_SERIES = [
    ("COCO only, 46k", "slots-rank64"),
    ("COCO + Carbon, r64", "slots-teacher-coco-rank64"),
    ("COCO + Carbon, r128", "slots-teacher-coco-rank128"),
    ("Carbon only, 17.7k", "slots-teacher-only-rank64"),
    ("+ prompts", "slots-prompted-teacher-coco-rank64"),
    ("+ size weighting", "slots-sizeweight-teacher-coco-rank64"),
    ("no denoising", "slots-nodenoise-teacher-coco-rank64"),
    ("O365 human for Carbon", "slots-o365human-coco-rank64"),
    ("COCO + human OI, O365", "slots-human-coco-rank64"),
    ("human OI, O365 only", "slots-human-only-rank64"),
    ("COCO 23k, human", "slots-coco23k-human-rank64"),
    ("COCO 23k, Carbon", "slots-coco23k-carbon-rank64"),
    ("COCO 23k, human + Carbon", "slots-coco23k-merged-rank64"),
    ("+ tile crops in training", "slots-tilecrops-teacher-coco-rank64"),
]
SELECTION_MODES = (("fast", None), ("tiled", 4))


def selection_rows(artifacts):
    rows = []
    for label, run in SELECTION_SERIES:
        path = artifacts / f"{run}-selection.json"
        if not path.exists():
            continue
        results = json.loads(path.read_text())["results"]
        modes = {}
        for mode, tiles in SELECTION_MODES:
            for r in results:
                s = r["setting"]
                if (s["tile_max_cells"], s["merge"], s["duplicate_iou"], s["box_power"]) == (
                    tiles,
                    "fuse",
                    0.5,
                    0.25,
                ):
                    modes[mode] = r["tests"]
        if len(modes) == len(SELECTION_MODES):
            rows.append((label, modes))
    return rows


def selection_area_figure(artifacts, out):
    """Area under the precision-recall curve and per-category AP for every head under the
    current selection, fast and tiled, on COCO and VOC."""
    rows = selection_rows(artifacts)
    panels = [
        ("COCO val2017: pooled area under the PR curve, IoU 0.5", 0, "pooled_area"),
        ("VOC 2007 test: pooled area under the PR curve, IoU 0.5", 1, "pooled_area"),
        ("COCO val2017: per-category AP, IoU 0.5 to 0.95", 0, "category_ap_iou_0.5_to_0.95"),
        ("VOC 2007 test: per-category AP, IoU 0.5 to 0.95", 1, "category_ap_iou_0.5_to_0.95"),
    ]
    figure, axes = plt.subplots(4, 1, figsize=(14, 15), facecolor=SURFACE)
    places = np.arange(len(rows))
    for axis, (title, test, key) in zip(axes, panels, strict=True):
        style(axis, title)
        for offset, (mode, _) in zip((-0.2, 0.2), SELECTION_MODES, strict=True):
            values = [modes[mode][test][key] for _, modes in rows]
            tiled = mode == "tiled"
            bars = axis.bar(
                places + offset,
                values,
                0.38,
                color=SURFACE if tiled else RED,
                edgecolor=RED,
                hatch="//" if tiled else None,
                linewidth=1.3,
                label=mode,
            )
            axis.bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=7, padding=2)
        best = max(m[mode][test][key] for _, m in rows for mode, _ in SELECTION_MODES)
        axis.set_ylim(0, best * 1.15)
        axis.set_xticks(places, [label for label, _ in rows], fontsize=8, color=MUTED)
        axis.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right", ncols=2)
    axes[-1].annotate(
        "Moondream gives no confidence scores, so it has no curve or area.",
        (0.0, -0.16),
        xycoords="axes fraction",
        fontsize=8,
        color=MUTED,
    )
    figure.tight_layout()
    figure.savefig(out / "selection-areas.png", dpi=180, facecolor=SURFACE)
    plt.close(figure)


# Moondream's single operating point per dataset (F1 at IoU 0.5), for reference lines.
MOONDREAM_F1 = {
    "coco": (
        ("Moondream 2", "moondream2-001", "coco_val2017"),
        ("Moondream 3.1", "moondream31-001", "coco_val2017"),
    ),
    "voc": (
        ("Moondream 2", "moondream2-photon-voc", "voc2007_test"),
        ("Moondream 3.1", "moondream31-voc", "voc2007_test"),
    ),
}


def selection_f1_figure(artifacts, out):
    """F1 at IoU 0.5 of every slot head under the current selection, fast and tiled, on
    COCO and VOC, each at the threshold tuned on COCO, against Moondream's."""
    rows = selection_rows(artifacts)
    figure, axes = plt.subplots(2, 1, figsize=(14, 8.5), facecolor=SURFACE)
    places = np.arange(len(rows))
    for axis, (test, name, title) in zip(
        axes,
        ((0, "coco", "COCO val2017, 500 images"), (1, "voc", "Pascal VOC 2007 test, 1,000 images")),
        strict=True,
    ):
        style(axis, f"{title}: F1 at IoU 0.5 at the threshold tuned on COCO")
        for offset, (mode, _) in zip((-0.2, 0.2), SELECTION_MODES, strict=True):
            values = [modes[mode][test]["iou_0.5"]["f1"] for _, modes in rows]
            tiled = mode == "tiled"
            bars = axis.bar(
                places + offset,
                values,
                0.38,
                color=SURFACE if tiled else RED,
                edgecolor=RED,
                hatch="//" if tiled else None,
                linewidth=1.3,
                label=mode,
            )
            axis.bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=7, padding=2)
        for (label, run, dataset), colour in zip(MOONDREAM_F1[name], (BLUE, YELLOW), strict=True):
            path = artifacts / run / "predictions.jsonl"
            if not path.exists():
                continue
            records = [r for r in read_records(path) if r["dataset"] == dataset]
            f1 = summarize(records)[dataset]["detection"]["iou_0.5"]["f1"]
            axis.axhline(
                f1, color=colour, linewidth=1.6, linestyle="--", label=f"{label}: {f1:.3f}"
            )
        values = [m[mode][test]["iou_0.5"]["f1"] for _, m in rows for mode, _ in SELECTION_MODES]
        axis.set_ylim(min(values) - 0.03, max(values) + 0.025)
        axis.set_xticks(places, [label for label, _ in rows], fontsize=8, color=MUTED)
        axis.legend(
            frameon=False,
            fontsize=8,
            labelcolor=INK,
            loc="lower right",
            bbox_to_anchor=(1, 1.0),
            ncols=4,
        )
    figure.tight_layout()
    figure.savefig(out / "selection-f1.png", dpi=180, facecolor=SURFACE)
    plt.close(figure)


# Precision-recall curves under the current selection: the approaches compared against
# Moondream's single operating points. (label, run, colour token index)
CURVE_SERIES = [
    ("COCO only, 46k", "slots-rank64", 3),
    ("COCO + Carbon, rank 64", "slots-teacher-coco-rank64", 1),
    ("COCO + Carbon, rank 128", "slots-teacher-coco-rank128", 2),
]
CURVE_COLOURS = {1: BLUE, 2: ORANGE, 3: AQUA, 4: YELLOW, 5: GREY, 6: VIOLET}


def selection_pr_figure(artifacts, out, images):
    """Precision against recall at IoU 0.5 of each approach under the current selection,
    without and with tiles, on COCO and VOC, with Moondream's operating points; each
    head's dot is its threshold tuned on COCO. `images` holds the evaluation images
    (for their sizes)."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from study_selection import apply, image_sizes, load

    configs = [
        Path(__file__).parent.parent / "configs" / n for n in ("slots-coco.json", "voc-slots.json")
    ]
    sizes = image_sizes(configs, images)
    aliases = {"voc2007_test": {"car": ["car", "truck"]}}
    figure, axes = plt.subplots(2, 2, figsize=(15, 12.5), facecolor=SURFACE)
    panels = (
        ("coco", "coco_val2017", "COCO val2017"),
        ("voc", "voc2007_test", "Pascal VOC 2007 test"),
    )
    for row, (split, dataset, title) in enumerate(panels):
        for col, (mode, tiles) in enumerate(SELECTION_MODES):
            axis = axes[row][col]
            style(axis, f"{title}, {'with 2x2 tiles' if tiles else 'full image only'}")
            axis.grid(axis="x", color=GRID, linewidth=0.8)
            for label, run, colour in CURVE_SERIES:
                path = artifacts / f"{run}-{split}-tiles2" / "predictions.jsonl"
                study = artifacts / f"{run}-selection.json"
                if not path.exists() or not study.exists():
                    continue
                setting = {
                    "tile_max_cells": tiles,
                    "merge": "fuse",
                    "duplicate_iou": 0.5,
                    "box_power": 0.25,
                }
                tuned = [
                    r for r in json.loads(study.read_text())["results"] if r["setting"] == setting
                ]
                records = apply(load(path), setting, aliases, sizes)
                precision, recall, _ = ranked_curve(records)
                ap = ranked_ap(records)["pooled_ap"]
                axis.plot(
                    recall,
                    precision,
                    color=CURVE_COLOURS[colour],
                    linewidth=1.8,
                    label=f"{label}, area {ap:.3f}",
                )
                if tuned:
                    point = summarize(above(records, tuned[0]["threshold"]))[dataset]["detection"][
                        "iou_0.5"
                    ]
                    axis.scatter(
                        [point["recall"]],
                        [point["precision"]],
                        s=45,
                        color=CURVE_COLOURS[colour],
                        edgecolor=SURFACE,
                        linewidth=1.5,
                        zorder=3,
                    )
            for name, run, md_dataset in MOONDREAM_F1[split]:
                path = artifacts / run / "predictions.jsonl"
                if not path.exists():
                    continue
                records = [r for r in read_records(path) if r["dataset"] == md_dataset]
                point = summarize(records)[md_dataset]["detection"]["iou_0.5"]
                axis.scatter(
                    [point["recall"]],
                    [point["precision"]],
                    s=130,
                    marker="D",
                    color=INK if name == "Moondream 3.1" else MUTED,
                    edgecolor=SURFACE,
                    linewidth=2,
                    zorder=4,
                    label=f"{name} (one point, F1 {point['f1']:.3f})",
                )
            axis.set_xlabel("Recall", fontsize=9, color=MUTED)
            axis.set_ylabel("Precision", fontsize=9, color=MUTED)
            low = 0.4 if split == "coco" else 0.6
            axis.set_xlim(low, 0.8 if split == "coco" else 1.0)
            axis.set_ylim(0.4 if split == "coco" else 0.6, 1.0)
            axis.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    axes[0][1].annotate(
        "Dots: each head at its threshold tuned on COCO. Diamonds: Moondream, which gives no "
        "confidence and so one point.",
        (0.99, 0.99),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    figure.tight_layout()
    figure.savefig(out / "selection-precision-recall.png", dpi=170, facecolor=SURFACE)
    plt.close(figure)


# The paper's architecture figure shows one VOC image going in and the slot head's
# detections coming out, each in the colour of its slot (the colours are repeated in
# paper/main.tex). (evaluation run, image id)
ARCHITECTURE_EXAMPLE = ("slots-teacher-coco-rank128-voc-final", "000666")
SLOT_COLOURS = (BLUE, ORANGE, AQUA, VIOLET)


def architecture_images(artifacts, out):
    """`architecture-input.jpg` and `architecture-output.jpg`: the example image, and the
    same image with the head's boxes, coloured slot by slot in descending score."""
    from PIL import ImageDraw

    run, image_id = ARCHITECTURE_EXAMPLE
    records = by_image(artifacts / run / "predictions.jsonl", "voc2007_test")[image_id]
    image = example_image("voc2007_test", image_id, out / "images")
    found = sorted(
        (
            (score, r["phrase"], box)
            for r in records
            for box, score in zip(r["predicted"], r["scores"], strict=True)
        ),
        reverse=True,
    )
    image.save(out / "architecture-input.jpg", quality=92)
    drawn = image.copy()
    pen = ImageDraw.Draw(drawn)
    width = max(3, round(max(image.size) / 120))
    for (_, label, box), colour in zip(found, SLOT_COLOURS, strict=False):
        pen.rectangle(box, outline=colour, width=width)
    drawn.save(out / "architecture-output.jpg", quality=92)
    return [label for _, label, _ in found]


def by_image(path, dataset):
    grouped = defaultdict(list)
    for record in rows(path):
        if record["dataset"] == dataset:
            grouped[record["image_id"]].append(record)
    return grouped


def examples_figure(artifacts, out, name, objects, skip=0, count=4, dataset="coco_val2017"):
    """Side-by-side boxes on `count` images whose annotated object count is in `objects`,
    chosen among the images every panel's evaluation covers."""
    panels = EXAMPLES[dataset]
    runs = {}
    for _, run in panels:
        if run:
            path = artifacts / run / "predictions.jsonl"
            if not path.exists():
                return
            runs[run] = by_image(path, dataset)
    reference = next(iter(runs.values()))
    low, high = objects
    chosen = [
        image_id
        for image_id, records in sorted(reference.items())
        if low <= sum(len(r["boxes"]) for r in records) <= high
        and all(image_id in found for found in runs.values())
    ][skip : skip + count]
    figure, axes = plt.subplots(
        count, len(panels), figsize=(13, 3.1 * count), facecolor=SURFACE, squeeze=False
    )
    for row, image_id in enumerate(chosen):
        image = example_image(dataset, image_id, out / "images")
        for axis, (title, run) in zip(axes[row], panels, strict=True):
            records = runs[run][image_id] if run else reference[image_id]
            boxes = [b for r in records for b in (r["predicted"] if run else r["boxes"])]
            axis.imshow(image)
            for x0, y0, x1, y1 in boxes:
                for colour, width in ((SURFACE, 3.5), (BLUE if run else INK, 1.8)):
                    axis.add_patch(
                        Rectangle(
                            (x0, y0),
                            x1 - x0,
                            y1 - y0,
                            fill=False,
                            edgecolor=colour,
                            linewidth=width,
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
    voc_figure(args.artifacts, args.out)
    selection_area_figure(args.artifacts, args.out)
    selection_f1_figure(args.artifacts, args.out)
    selection_pr_figure(args.artifacts, args.out, args.out / "images")
    architecture_images(args.artifacts, args.out)
    area_figure(args.artifacts, args.out)
    examples_figure(args.artifacts, args.out, "examples.png", objects=(4, 9))
    examples_figure(args.artifacts, args.out, "examples-2.png", objects=(4, 9), skip=4)
    examples_figure(args.artifacts, args.out, "examples-crowded.png", objects=(12, 25))
    examples_figure(
        args.artifacts, args.out, "examples-voc.png", objects=(3, 8), dataset="voc2007_test"
    )
    examples_figure(
        args.artifacts,
        args.out,
        "examples-mot.png",
        objects=(8, 40),
        count=3,
        dataset="mot17_train",
    )


if __name__ == "__main__":
    main()
