"""Scores and latency summaries from per-query prediction records.

A record is a query (see `tasks`) plus "predicted" boxes in pixels, "seconds" for
that phrase, and the sample's "shared_seconds" (work done once per image).
Detections carry no confidence, so the scores are precision, recall and F1 at fixed
IoU thresholds, plus the unit-confidence average precision Moondream reports.
"""

from collections import defaultdict

import numpy as np

from parallel_detection.boxes import areas, iou_matrix, match

IOU_THRESHOLDS = (0.5, 0.75)
# COCO's size classes, by truth-box area in pixels of the original image.
SIZES = (("small", 0, 32**2), ("medium", 32**2, 96**2), ("large", 96**2, float("inf")))
COUNT_BUCKETS = (("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21+", 21, float("inf")))
EMITTED_BUCKETS = (("0", 0, 0),) + COUNT_BUCKETS


def bucket(value, buckets):
    return next(name for name, low, high in buckets if low <= value <= high)


def size_class(area):
    return next(name for name, low, high in SIZES if low <= area < high)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def prf(tp, fp, fn):
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def unit_confidence_ap(predicted, truth, threshold):
    """Average precision as Moondream's COCO script computes it for one phrase.

    Every box has confidence 1, so boxes are taken in emitted order; each takes the
    unmatched truth box it overlaps most. Precision is interpolated at 11 recall points.
    """
    iou = iou_matrix(predicted, truth)
    taken = np.zeros(len(truth), dtype=bool)
    hits = np.zeros(len(predicted))
    for index in range(len(predicted)):
        overlaps = np.where(taken, -1.0, iou[index])
        best = int(overlaps.argmax())
        if overlaps[best] >= threshold and overlaps[best] > 0:
            hits[index], taken[best] = 1, True
    found = np.cumsum(hits)
    recalls = found / len(truth)
    precisions = found / np.arange(1, len(predicted) + 1)
    points = [precisions[recalls >= t].max(initial=0.0) for t in np.arange(0, 1.1, 0.1)]
    return float(np.mean(points))


def detection_summary(records):
    """Micro-averaged scores over phrases that name something in the image.

    Boxes for an "absent" phrase count as false positives.
    """
    present = [r for r in records if r["kind"] != "absent"]
    absent = [r for r in records if r["kind"] == "absent"]
    out = {"queries": len(present), "truth_boxes": sum(len(r["boxes"]) for r in present)}
    for threshold in IOU_THRESHOLDS:
        tp = fp = fn = phrases_hit = 0
        hit = defaultdict(
            lambda: defaultdict(lambda: [0, 0])
        )  # breakdown -> bucket -> [found, total]
        for r in present:
            found = {t for _, t in match(r["predicted"], r["boxes"], threshold)}
            tp += len(found)
            fn += len(r["boxes"]) - len(found)
            phrases_hit += bool(found)
            if r["exhaustive"]:
                fp += len(r["predicted"]) - len(found)
            for index, area in enumerate(areas(r["boxes"])):
                for breakdown, key in (
                    ("size", size_class(area)),
                    ("instances_in_query", bucket(len(r["boxes"]), COUNT_BUCKETS)),
                    ("group", r["group"]),
                ):
                    hit[breakdown][key][0] += index in found
                    hit[breakdown][key][1] += 1
        fp += sum(len(r["predicted"]) for r in absent)
        aps = [unit_confidence_ap(r["predicted"], r["boxes"], threshold) for r in present]
        out[f"iou_{threshold}"] = (
            prf(tp, fp, fn)
            | {
                # Share of phrases with at least one correct box.
                "phrase_hit_rate": ratio(phrases_hit, len(present)),
                "unit_confidence_map": float(np.mean(aps)) if aps else None,
            }
            | {
                f"recall_by_{breakdown}": {
                    key: {"recall": ratio(found, total), "truth_boxes": total}
                    for key, (found, total) in sorted(buckets.items())
                }
                for breakdown, buckets in hit.items()
            }
        )
    if absent:
        out["absent"] = {
            "queries": len(absent),
            "false_alarm_rate": ratio(sum(bool(r["predicted"]) for r in absent), len(absent)),
            "boxes_per_query": ratio(sum(len(r["predicted"]) for r in absent), len(absent)),
        }
    return out


def referring_summary(records):
    """Accuracy of the first predicted box against the single referred object."""
    records = [r for r in records if r["kind"] == "referring"]
    out = {"queries": len(records)}
    for threshold in IOU_THRESHOLDS:
        correct = sum(
            bool(r["predicted"]) and iou_matrix(r["predicted"][:1], r["boxes"])[0, 0] >= threshold
            for r in records
        )
        out[f"accuracy_iou_{threshold}"] = ratio(int(correct), len(records))
    out["one_box_rate"] = ratio(sum(len(r["predicted"]) == 1 for r in records), len(records))
    out["no_box_rate"] = ratio(sum(not r["predicted"] for r in records), len(records))
    return out


def spread(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    p50, p90 = np.percentile(values, [50, 90])
    return {
        "mean": float(values.mean()),
        "p50": float(p50),
        "p90": float(p90),
        "n": int(values.size),
    }


def latency_summary(records):
    """Seconds per image and per phrase, and how phrase time grows with boxes emitted."""
    per_image = defaultdict(lambda: {"shared": 0.0, "phrases": 0.0, "boxes": 0, "queries": 0})
    for r in records:
        image = per_image[r["image_id"]]
        image["shared"] = r["shared_seconds"]
        image["phrases"] += r["seconds"]
        image["boxes"] += len(r["predicted"])
        image["queries"] += 1
    emitted = np.array([len(r["predicted"]) for r in records], dtype=np.float64)
    seconds = np.array([r["seconds"] for r in records], dtype=np.float64)
    by_emitted = defaultdict(list)
    for count, value in zip(emitted, seconds, strict=True):
        by_emitted[bucket(count, EMITTED_BUCKETS)].append(value)
    out = {
        "image_seconds": spread([i["shared"] + i["phrases"] for i in per_image.values()]),
        "shared_seconds": spread([i["shared"] for i in per_image.values()]),
        "phrase_seconds": spread(seconds),
        "phrases_per_image": spread([i["queries"] for i in per_image.values()]),
        "boxes_per_image": spread([i["boxes"] for i in per_image.values()]),
        "phrase_seconds_by_boxes_emitted": {
            name: spread(by_emitted[name]) for name, _, _ in EMITTED_BUCKETS if by_emitted[name]
        },
    }
    parts = {r["image_id"]: r["components"] for r in records if r.get("components")}
    if parts:
        # Detectors that time their stages report where the shared seconds go.
        out["shared_components"] = {
            name: spread([p[name] for p in parts.values()]) for name in next(iter(parts.values()))
        }
    if len(set(emitted.tolist())) > 1:
        slope, intercept = np.polyfit(emitted, seconds, 1)
        out["phrase_seconds_linear_fit"] = {
            "seconds_per_box": float(slope),
            "seconds_at_zero_boxes": float(intercept),
        }
    return out


def summarize(records):
    """Per-dataset summaries; `records` hold a "dataset" name each."""
    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)
    out = {}
    for name, group in sorted(by_dataset.items()):
        out[name] = {"latency": latency_summary(group), "detection": detection_summary(group)}
        if any(r["kind"] == "referring" for r in group):
            out[name]["referring"] = referring_summary(group)
    return out


def agreement(reference, records):
    """How closely two runs over the same queries agree, box by box in emitted order."""
    same_count = identical = 0
    overlaps = []
    for a, b in zip(reference, records, strict=True):
        if (a["dataset"], a["image_id"], a["phrase"]) != (b["dataset"], b["image_id"], b["phrase"]):
            raise ValueError("Runs cover different queries")
        if len(a["predicted"]) != len(b["predicted"]):
            continue
        same_count += 1
        paired = iou_matrix(a["predicted"], b["predicted"]).diagonal()
        overlaps += paired.tolist()
        identical += a["predicted"] == b["predicted"]
    return {
        "queries": len(reference),
        "same_box_count_rate": ratio(same_count, len(reference)),
        "identical_boxes_rate": ratio(identical, len(reference)),
        "paired_iou": spread(overlaps),
    }
