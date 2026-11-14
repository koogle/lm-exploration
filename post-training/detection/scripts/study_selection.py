"""Compare ways of choosing slots: tiles, duplicate merging, score and label aliases.

    python scripts/study_selection.py --tune artifacts/HEAD-coco-tune-tiles2/predictions.jsonl \
        --test artifacts/HEAD-coco-tiles2/predictions.jsonl \
        --test artifacts/HEAD-voc-tiles2/predictions.jsonl --config configs/slots-coco-tune.json \
        --config configs/slots-coco.json --config configs/voc-slots.json \
        > artifacts/HEAD-selection.json

Reads evaluations made with `--all-slots` (and `--tiles` for tile candidates), and the
evaluation configs (`--config`, one or more) for image sizes. For each
setting it rescores every candidate, applies `merge.select` exactly as the detector
does, chooses the threshold with the best F1 on the tuning evaluation, and reports on
each test evaluation: precision, recall and F1 at IoU 0.5 and 0.75, recall by object
size in grid cells, and the pooled area under the precision-recall curve. Aliases from
`--aliases` (JSON: phrase -> head labels) apply to the evaluations whose dataset they
name.
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from parallel_detection.boxes import match
from parallel_detection.merge import cells, select
from parallel_detection.metrics import best_threshold, ranked_ap, summarize
from parallel_detection.run import above, read_records

CELL_BINS = (("under 1 cell", 0, 1), ("1 to 4 cells", 1, 4), ("4 cells and over", 4, 1e9))


def load(path):
    """Per image: its records and its candidates."""
    images = defaultdict(lambda: {"records": [], "slots": None})
    for r in read_records(path):
        images[r["image_id"]]["records"].append(r)
        if "slots" in r:
            images[r["image_id"]]["slots"] = r["slots"]
    return dict(images)


def image_sizes(configs, folder):
    """(dataset, image id) -> (width, height), from the evaluation configs' samples."""
    from parallel_detection.run import build_samples

    sizes = {}
    for path in configs:
        for sample in build_samples(json.loads(Path(path).read_text()), folder):
            sizes[sample["dataset"], sample["image_id"]] = (sample["width"], sample["height"])
    return sizes


def apply(images, setting, aliases, sizes):
    power = setting["box_power"]
    out = []
    for image in images.values():
        first = image["records"][0]
        width, height = sizes[first["dataset"], first["image_id"]]
        candidates = [
            s | {"score": s["objectness"] * s["label_confidence"] * s["box_confidence"] ** power}
            for s in image["slots"]
        ]
        phrases = [r["phrase"] for r in image["records"]]
        boxes, scores = select(
            candidates,
            phrases,
            width,
            height,
            threshold=0.0,
            duplicate_iou=setting["duplicate_iou"],
            merge=setting["merge"],
            aliases=aliases.get(first["dataset"]),
            tile_max_cells=setting["tile_max_cells"],
        )
        for r, b, s in zip(image["records"], boxes, scores, strict=True):
            out.append(r | {"predicted": b, "scores": s, "size": (width, height)})
    return out


def recall_by_cells(records):
    hit = defaultdict(lambda: [0, 0])
    for r in records:
        found = {t for _, t in match(r["predicted"], r["boxes"], 0.5)}
        for index, box in enumerate(r["boxes"]):
            size = cells(box, *r["size"])
            name = next(n for n, low, high in CELL_BINS if low <= size < high)
            hit[name][0] += index in found
            hit[name][1] += 1
    return {name: found / total for name, (found, total) in hit.items()}


def report(records, threshold):
    dataset = records[0]["dataset"]
    kept = above(records, threshold)
    detection = summarize(kept)[dataset]["detection"]
    return {
        "dataset": dataset,
        "iou_0.5": {k: detection["iou_0.5"][k] for k in ("precision", "recall", "f1")},
        "f1_iou_0.75": detection["iou_0.75"]["f1"],
        "recall_by_cells": recall_by_cells(kept),
        "pooled_area": ranked_ap(records)["pooled_ap"],
    }


def settings():
    yield from (
        {"tile_max_cells": t, "merge": m, "duplicate_iou": d, "box_power": p}
        for t, m, d, p in itertools.product(
            (None, 1, 2, 4), ("nms", "fuse"), (0.5, 0.7), (1.0, 0.25)
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tune", required=True)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--images", default="artifacts/images", help="sample images folder")
    parser.add_argument("--aliases", default="{}", help='{"dataset": {"phrase": [labels]}}')
    args = parser.parse_args()
    aliases = json.loads(args.aliases)
    sizes = image_sizes(args.config, args.images)
    tune, tests = load(args.tune), [load(path) for path in args.test]
    grid = np.unique(np.concatenate([np.geomspace(1e-3, 0.05, 30), np.arange(0.05, 0.95, 0.02)]))
    results = []
    for setting in settings():
        tuned = apply(tune, setting, aliases, sizes)
        threshold = best_threshold(tuned, tuned[0]["dataset"], grid)
        f1 = summarize(above(tuned, threshold))[tuned[0]["dataset"]]["detection"]["iou_0.5"]["f1"]
        results.append(
            {
                "setting": setting,
                "threshold": threshold,
                "tune_f1": f1,
                "tests": [
                    report(apply(test, setting, aliases, sizes), threshold) for test in tests
                ],
            }
        )
    print(
        json.dumps(
            {"tune": args.tune, "tests": args.test, "aliases": aliases, "results": results},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
