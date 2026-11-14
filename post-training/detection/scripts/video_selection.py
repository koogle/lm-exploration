"""Hysteresis and tracking on video, scored frame by frame.

    python scripts/video_selection.py --config configs/mot-slots.json \
        --predictions artifacts/slots-teacher-coco-rank128-mot-tiles2/predictions.jsonl \
        --images /tmp/claude-0/sample-images > artifacts/mot-hysteresis.json

Reads a MOT17 evaluation that recorded every candidate (`--all-slots --tiles 2`). For
each sequence it applies `merge.select` frame by frame in order, passing the boxes kept
in the previous sampled frame as `previous`, so a candidate continuing one of them
passes from `keep_threshold` (a share of the threshold). Thresholds are the ones tuned
on COCO (`MODES`); nothing is tuned on the video. Optionally the kept boxes then pass
through `tracking.Tracker` (`TRACKERS`: smoothing and coasting), whose tracked boxes are
what is scored. Reports per mode, keep share and tracker: precision, recall and F1 at
IoU 0.5, and `video_stability.stability`.
"""

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from study_selection import image_sizes, load
from video_stability import stability, track_ids

from parallel_detection.merge import select
from parallel_detection.metrics import summarize
from parallel_detection.tracking import Tracker

# (name, tile_max_cells, threshold tuned on COCO for slots-teacher-coco-rank128)
MODES = (("fast", None, 0.31), ("tiled", 4, 0.33))
KEEP_SHARES = (1.0, 0.5)
# (name, Tracker arguments or None)
TRACKERS = (
    ("none", None),
    ("smooth", {"smoothing": 0.5, "coast": 0}),
    ("coast 1", {"smoothing": 0.0, "coast": 1}),
    ("smooth, coast 1", {"smoothing": 0.5, "coast": 1}),
    ("smooth, coast 2", {"smoothing": 0.5, "coast": 2}),
)


def run(images, sizes, tiles, threshold, keep_share, tracker=None):
    by_sequence = defaultdict(list)
    for image_id in images:
        sequence, frame = image_id.split("/")
        by_sequence[sequence].append((int(frame), image_id))
    out = []
    for sequence, frames in by_sequence.items():
        previous = None
        tracks = Tracker(**tracker) if tracker is not None else None
        for _, image_id in sorted(frames):
            image = images[image_id]
            first = image["records"][0]
            width, height = sizes[first["dataset"], image_id]
            candidates = [
                c | {"score": c["objectness"] * c["label_confidence"] * c["box_confidence"] ** 0.25}
                for c in image["slots"]
            ]
            phrases = [r["phrase"] for r in image["records"]]
            boxes, scores = select(
                candidates,
                phrases,
                width,
                height,
                threshold,
                duplicate_iou=0.5,
                merge="fuse",
                tile_max_cells=tiles,
                previous=previous,
                keep_threshold=threshold * keep_share,
            )
            previous = dict(zip(phrases, boxes, strict=True))
            if tracks is not None:
                flat = [
                    (p, b, s)
                    for p, bs, ss in zip(phrases, boxes, scores, strict=True)
                    for b, s in zip(bs, ss, strict=True)
                ]
                tracked = tracks.update(
                    [b for _, b, _ in flat], [s for _, _, s in flat], [p for p, _, _ in flat]
                )
                boxes = [[t["box"] for t in tracked if t["label"] == p] for p in phrases]
                scores = [[t["score"] for t in tracked if t["label"] == p] for p in phrases]
            for r, b, s in zip(image["records"], boxes, scores, strict=True):
                out.append(r | {"predicted": b, "scores": s})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--images", required=True)
    args = parser.parse_args()
    (dataset,) = json.loads(Path(args.config).read_text())["datasets"]
    sizes = image_sizes([args.config], args.images)
    images = load(args.predictions)
    ids = track_ids(dataset)
    results = []
    for (mode, tiles, threshold), share, (tracking, tracker) in itertools.product(
        MODES, KEEP_SHARES, TRACKERS
    ):
        records = run(images, sizes, tiles, threshold, share, tracker)
        detection = summarize(records)[dataset["name"]]["detection"]["iou_0.5"]
        results.append(
            {
                "mode": mode,
                "threshold": threshold,
                "keep_share": share,
                "tracker": tracking,
                "iou_0.5": {k: detection[k] for k in ("precision", "recall", "f1")},
                "stability": stability(records, ids),
            }
        )
    print(json.dumps({"predictions": args.predictions, "results": results}, indent=2))


if __name__ == "__main__":
    main()
