"""Figures from downloaded run outputs.

    python scripts/figures.py --artifacts artifacts --out artifacts/figures

Reads `<artifacts>/<run>/heldout.jsonl` for the training runs and
`<artifacts>/<run>/summary.json` and `predictions.jsonl` for the COCO evaluations, and
writes `training.png`, `coco.png` and `examples.png`.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from parallel_detection.run import fetch_image

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e4e3dd"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TRAINING = [
    ("slots-001", "Regressed boxes, 11k images", BLUE),
    ("slots-bins-4", "Coordinate bins, 11k images", ORANGE),
    ("slots-bins-16", "Coordinate bins, 44k images", AQUA),
]
EVALUATIONS = [
    ("moondream2-001", "Moondream 2 detect (L4)", BLUE),
    ("slots-001-coco", "Slot head, regressed boxes (L4)", ORANGE),
    ("slots-bins-4-coco", "Slot head, coordinate bins (L4)", AQUA),
    ("moondream31-001", "Moondream 3.1, Photon (L40S)", YELLOW),
]
EXAMPLES = [
    ("Annotated", None),
    ("Moondream 2", "moondream2-001"),
    ("Moondream 3.1", "moondream31-001"),
    ("Slot head, bins", "slots-bins-4-coco"),
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


def training_figure(artifacts, out):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    panels = [
        (
            "Held-out F1 at IoU 0.5 (best objectness threshold)",
            lambda r: max(r["f1_iou_0.5"].values()),
        ),
        ("Held-out IoU of matched boxes", lambda r: r["matched_iou"]),
    ]
    for axis, (title, value) in zip(axes, panels, strict=True):
        style(axis, title)
        for run, label, color in TRAINING:
            path = artifacts / run / "heldout.jsonl"
            if not path.exists():
                continue
            points = rows(path)
            minutes = [r["seconds"] / 60 for r in points]
            values = [value(r) for r in points]
            axis.plot(
                minutes, values, color=color, linewidth=2, marker="o", markersize=4, label=label
            )
            axis.annotate(
                f"{values[-1]:.2f}",
                (minutes[-1], values[-1]),
                textcoords="offset points",
                xytext=(6, 0),
                va="center",
                fontsize=9,
                color=INK,
            )
        axis.set_xlabel("Training minutes on one H100", fontsize=9, color=MUTED)
        axis.set_ylim(0, 0.85)
        axis.margins(x=0.08)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    figure.tight_layout()
    figure.savefig(out / "training.png", dpi=200, facecolor=SURFACE)
    plt.close(figure)


def coco_figure(artifacts, out):
    figure, axis = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    style(axis, "COCO val2017, 500 images: F1 at IoU 0.5 against seconds per image")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    for run, label, color in EVALUATIONS:
        path = artifacts / run / "summary.json"
        if not path.exists():
            continue
        coco = json.loads(path.read_text())["datasets"]["coco_val2017"]
        seconds = coco["latency"]["image_seconds"]["mean"]
        f1 = coco["detection"]["iou_0.5"]["f1"]
        axis.scatter([seconds], [f1], s=90, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        axis.annotate(
            f"{label}\n{f1:.2f} F1, {seconds:.2f} s",
            (seconds, f1),
            textcoords="offset points",
            xytext=(10, -4),
            fontsize=9,
            color=INK,
        )
    axis.set_xlabel("Seconds per image (all prompted categories)", fontsize=9, color=MUTED)
    axis.set_xlim(0, 1.6)
    axis.set_ylim(0.35, 0.78)
    figure.tight_layout()
    figure.savefig(out / "coco.png", dpi=200, facecolor=SURFACE)
    plt.close(figure)


def by_image(path):
    grouped = defaultdict(list)
    for record in rows(path):
        if record["dataset"] == "coco_val2017":
            grouped[record["image_id"]].append(record)
    return grouped


def examples_figure(artifacts, out, count=4):
    runs = {run: by_image(artifacts / run / "predictions.jsonl") for _, run in EXAMPLES if run}
    reference = runs["moondream2-001"]
    # The first images, by id, with a handful of annotated objects.
    chosen = [
        image_id
        for image_id, records in sorted(reference.items())
        if 4 <= sum(len(r["boxes"]) for r in records) <= 9 and len(records) >= 2
    ][:count]
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
                caption += ": " + ", ".join(r["phrase"] for r in records)
            axis.set_title(
                f"{title}\n{caption}" if row == 0 else caption, fontsize=9, color=INK, loc="left"
            )
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(out / "examples.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    training_figure(args.artifacts, args.out)
    coco_figure(args.artifacts, args.out)
    examples_figure(args.artifacts, args.out)


if __name__ == "__main__":
    main()
