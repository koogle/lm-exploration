"""Where a slot head's misses come from, and how re-scoring its slots moves the curve.

    python scripts/diagnose_slots.py --predictions artifacts/RUN-coco-v2/predictions.jsonl \
        --threshold 0.0133 > artifacts/RUN-coco-v2/diagnosis.json
    python scripts/diagnose_slots.py --predictions artifacts/A-coco-v2/predictions.jsonl \
        --threshold 0.0133 --compare artifacts/B-coco-v2/predictions.jsonl \
        --compare-threshold 0.0152

Reads an evaluation made with `--all-slots` (every slot of every image recorded). For
each annotated object it records the first condition that holds: no slot box within
IoU 0.1, only loose boxes (IoU 0.1 to 0.5), boxed with a wrong or unknown label, boxed
and labelled but scored below the threshold, or found. `--score` names the slot score
the threshold applies to (by default the detector's own). It then ranks the same slots
under other confidence scores, with and without removing same-label duplicates, and
reports the area under the precision-recall curve, the best F1 and recall at fixed
precisions. `--compare` follows the same objects through a second head and counts how
each object's fate changes between the two, by object size.
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


DEFAULT_SCORE = "objectness x label x box confidence"


def object_fates(by_image, slots, threshold, score):
    """{(image, phrase, index): (size, fate)} for every annotated object, and the
    label confusions of boxed but mislabelled objects."""
    out = {}
    confusions = Counter()
    for image, queries in by_image.items():
        found = slots[image]
        boxes = [s["box"] for s in found]
        for query in queries:
            for index, (box, area) in enumerate(
                zip(query["boxes"], areas(query["boxes"]), strict=True)
            ):
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
                elif max(score(found[k]) for k in right) < threshold:
                    fate = "below threshold"
                else:
                    fate = "found"
                out[(image, query["phrase"], index)] = (size_class(area), fate)
    return out, confusions


def fates(by_image, slots, threshold, score):
    per_object, confusions = object_fates(by_image, slots, threshold, score)
    counts = defaultdict(Counter)
    for size, fate in per_object.values():
        counts[size][fate] += 1
    total = Counter()
    for by_fate in counts.values():
        total.update(by_fate)
    return {
        "objects": dict(total),
        "by_size": {size: dict(by_fate) for size, by_fate in counts.items()},
        "label_confusions": dict(confusions.most_common(10)),
    }


def transitions(first, second):
    """How each object's fate under the first head changes under the second, by size."""
    if first.keys() != second.keys():
        raise ValueError("The two evaluations cover different objects")
    counts = defaultdict(Counter)
    for key, (size, fate) in first.items():
        counts[size][f"{fate} -> {second[key][1]}"] += 1
        counts["all"][f"{fate} -> {second[key][1]}"] += 1
    return {size: dict(by_change.most_common()) for size, by_change in counts.items()}


def load(path, dataset):
    records = [r for r in read_records(path) if r["dataset"] == dataset]
    slots = {r["image_id"]: r["slots"] for r in records if "slots" in r}
    by_image = defaultdict(list)
    for r in records:
        by_image[r["image_id"]].append(r)
    return by_image, slots


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
    parser.add_argument("--score", choices=SCORES, default=DEFAULT_SCORE)
    parser.add_argument("--dataset", default="coco_val2017")
    parser.add_argument("--compare", help="a second head's predictions on the same objects")
    parser.add_argument("--compare-threshold", type=float)
    args = parser.parse_args()
    by_image, slots = load(args.predictions, args.dataset)
    score = SCORES[args.score]
    rescoring = {}
    for name, score in SCORES.items():
        rescoring[name] = curve_summary(rescored(by_image, slots, score))
        rescoring[f"{name}, duplicates removed"] = curve_summary(
            rescored(by_image, slots, score, args.duplicate_iou)
        )
    result = {
        "dataset": args.dataset,
        "score": args.score,
        "threshold": args.threshold,
        "duplicate_iou": args.duplicate_iou,
        "fates": fates(by_image, slots, args.threshold, score),
        "rescoring": rescoring,
    }
    if args.compare:
        other_images, other_slots = load(args.compare, args.dataset)
        threshold = args.compare_threshold
        result["compared_with"] = {
            "predictions": args.compare,
            "threshold": threshold,
            "fates": fates(other_images, other_slots, threshold, score),
            "transitions": transitions(
                object_fates(by_image, slots, args.threshold, score)[0],
                object_fates(other_images, other_slots, threshold, score)[0],
            ),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
