"""Where a slot head's misses come from, and how re-scoring its slots moves the curve.

    python scripts/diagnose_slots.py --predictions artifacts/RUN-allslots/predictions.jsonl \
        --threshold 0.9 > artifacts/RUN-allslots/diagnosis.json

Reads an evaluation made with `--all-slots` (every slot of every image recorded). For
each annotated object it records the first condition that holds: no slot box within
IoU 0.1, only loose boxes (IoU 0.1 to 0.5), boxed with a wrong or unknown label, boxed
and labelled but below the threshold, or found. It then ranks the same slots under
other confidence scores, with and without removing same-label duplicates, and reports
the area under the precision-recall curve, the best F1 and recall at fixed precisions.
"""

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from parallel_detection.boxes import iou_matrix
from parallel_detection.metrics import areas, ranked_ap, ranked_curve, size_class
from parallel_detection.run import read_records

SCORES = {
    "objectness": lambda s: s["objectness"],
    "objectness x label confidence": lambda s: s["objectness"] * s["label_confidence"],
    "objectness x label x box confidence": lambda s: (
        s["objectness"] * s["label_confidence"] * s["box_confidence"]
    ),
}
PRECISIONS = (0.8, 0.743, 0.682, 0.6)


def fates(by_image, slots, threshold):
    counts = defaultdict(Counter)
    confusions = Counter()
    for image, queries in by_image.items():
        found = slots[image]
        boxes = [s["box"] for s in found]
        for query in queries:
            for box, area in zip(query["boxes"], areas(query["boxes"]), strict=True):
                iou = iou_matrix(boxes, [box])[:, 0]
                near = np.flatnonzero(iou >= 0.5)
                right = [k for k in near if found[k]["label"] == query["phrase"]]
                if not len(near):
                    fate = "only loose boxes" if (iou >= 0.1).any() else "no box near it"
                elif not right:
                    top = max(near, key=lambda k: found[k]["objectness"])
                    label = found[top]["label"]
                    fate = "wrong label" if label is not None else "label not a category"
                    confusions[f"{query['phrase']} -> {label}"] += 1
                elif max(found[k]["objectness"] for k in right) < threshold:
                    fate = "below threshold"
                else:
                    fate = "found"
                counts[size_class(area)][fate] += 1
    total = Counter()
    for by_fate in counts.values():
        total.update(by_fate)
    return {
        "objects": dict(total),
        "by_size": {size: dict(by_fate) for size, by_fate in counts.items()},
        "label_confusions": dict(confusions.most_common(10)),
    }


def rescored(by_image, slots, score, duplicate_iou=None):
    out = []
    for image, queries in by_image.items():
        for query in queries:
            mine = sorted(
                (s for s in slots[image] if s["label"] == query["phrase"]), key=score, reverse=True
            )
            if duplicate_iou is not None:
                kept = []
                for s in mine:
                    if all(iou_matrix([s["box"]], [k["box"]])[0, 0] < duplicate_iou for k in kept):
                        kept.append(s)
                mine = kept
            out.append(
                query | {"predicted": [s["box"] for s in mine], "scores": [score(s) for s in mine]}
            )
    return out


def curve_summary(records):
    precision, recall, _ = ranked_curve(records)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return ranked_ap(records) | {
        "best_f1": float(f1.max()),
        "recall_at_precision": {
            str(p): float(recall[precision >= p].max()) if (precision >= p).any() else 0.0
            for p in PRECISIONS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--duplicate-iou", type=float, default=0.7)
    args = parser.parse_args()
    records = [r for r in read_records(args.predictions) if r["dataset"] == "coco_val2017"]
    slots = {r["image_id"]: r["slots"] for r in records if "slots" in r}
    by_image = defaultdict(list)
    for r in records:
        by_image[r["image_id"]].append(r)
    rescoring = {}
    for name, score in SCORES.items():
        rescoring[name] = curve_summary(rescored(by_image, slots, score))
        rescoring[f"{name}, duplicates removed"] = curve_summary(
            rescored(by_image, slots, score, args.duplicate_iou)
        )
    result = {
        "threshold": args.threshold,
        "duplicate_iou": args.duplicate_iou,
        "fates": fates(by_image, slots, args.threshold),
        "rescoring": rescoring,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
