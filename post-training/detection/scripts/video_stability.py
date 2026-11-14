"""How steadily detectors find the same people from one sampled video frame to the next.

    python scripts/video_stability.py --config configs/mot-slots.json \
        --predictions artifacts/moondream31-mot/predictions.jsonl \
        --predictions artifacts/slots-teacher-coco-rank128-mot-fast/predictions.jsonl \
        > artifacts/mot-stability.json

Reads evaluations of a MOT17 config (`tasks.mot_samples`: every n-th frame, people as
truth) and the sequences' ground truth, which gives each person a track id. For every
person annotated in two consecutive sampled frames it records whether each detector
found them (IoU 0.5, one-to-one) in each. Reports per detector and sequence:

- found: the share of people found in a frame;
- lost next frame: of the people found in a frame, the share not found in the next
  sampled frame although still annotated there (flicker);
- regained next frame: of the people missed, the share found in the next frame;
- box count change: mean absolute change in the number of boxes between frames.
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from parallel_detection.boxes import match
from parallel_detection.run import read_records
from parallel_detection.tasks import MOT_PERSON_CLASSES


def track_ids(dataset):
    """(sequence, frame) -> {rounded box: track id}, from the sequences' gt.txt."""
    from huggingface_hub import hf_hub_download

    out = defaultdict(dict)
    for sequence in dataset["sequences"]:
        path = hf_hub_download(
            dataset["repo"],
            f"{dataset['split']}/{sequence}/gt/gt.txt",
            repo_type="dataset",
            revision=dataset["revision"],
        )
        for line in Path(path).read_text().splitlines():
            frame, track, x, y, w, h, _, kind, _ = line.split(",")
            if int(kind) in MOT_PERSON_CLASSES:
                x0, y0 = float(x) - 1, float(y) - 1
                box = (x0, y0, x0 + float(w), y0 + float(h))
                out[sequence, int(frame)][tuple(round(v, 2) for v in box)] = int(track)
    return out


def stability(path, ids):
    frames = {}
    for r in read_records(path):
        sequence, frame = r["image_id"].split("/")
        found = {t for _, t in match(r["predicted"], r["boxes"], 0.5)}
        tracks = ids[sequence, int(frame)]
        frames[sequence, int(frame)] = (
            {
                tracks[tuple(round(v, 2) for v in box)]: i in found
                for i, box in enumerate(r["boxes"])
            },
            len(r["predicted"]),
        )
    out = {}
    for sequence in sorted({s for s, _ in frames}):
        order = sorted(f for s, f in frames if s == sequence)
        found = total = lost = kept = regained = missed = 0
        changes = []
        for a, b in itertools.pairwise(order):
            (now, count_a), (then, count_b) = frames[sequence, a], frames[sequence, b]
            changes.append(abs(count_b - count_a))
            for track, hit in now.items():
                found += hit
                total += 1
                if track not in then:
                    continue
                if hit:
                    kept += 1
                    lost += not then[track]
                else:
                    missed += 1
                    regained += then[track]
        out[sequence] = {
            "found": found / max(total, 1),
            "lost_next_frame": lost / max(kept, 1),
            "regained_next_frame": regained / max(missed, 1),
            "box_count_change": float(np.mean(changes)) if changes else 0.0,
            "pairs": kept + missed,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", action="append", required=True)
    args = parser.parse_args()
    (dataset,) = json.loads(Path(args.config).read_text())["datasets"]
    ids = track_ids(dataset)
    print(json.dumps({path: stability(path, ids) for path in args.predictions}, indent=2))


if __name__ == "__main__":
    main()
