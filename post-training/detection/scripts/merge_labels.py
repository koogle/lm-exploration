"""Merge two label files of the same images: every box of the first, plus the second's
boxes that no first box of the same label overlaps.

    python scripts/merge_labels.py --first artifacts/human-labels/coco-human.jsonl \
        --second artifacts/teacher-train-coco/google--gemini-4-carbon.jsonl \
        --out artifacts/human-labels/coco-human-plus-carbon.jsonl

Both files are in the teacher-label format (one JSON line per image). Only images in
both are written. A second box is added when its IoU with every first box of its label
is below `--iou`, so it is an object the first labels miss rather than a second box on
one they have.
"""

import argparse
import json
from pathlib import Path

from parallel_detection.merge import add_unmatched


def read(path):
    rows = map(json.loads, Path(path).read_text().splitlines())
    return {r["image_id"]: r for r in rows if "error" not in r}


def merge(first, second, iou):
    out = []
    for image_id, a in first.items():
        b = second.get(image_id)
        if b is None:
            continue
        objects = add_unmatched(a["objects"], b["objects"], iou)
        out.append(a | {"model": f"{a['model']}+{b['model']}", "objects": objects})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    first = read(args.first)
    rows = merge(first, read(args.second), args.iou)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    total = sum(len(r["objects"]) for r in rows)
    added = total - sum(len(first[r["image_id"]]["objects"]) for r in rows)
    print(json.dumps({"images": len(rows), "objects": total, "added": added}))


if __name__ == "__main__":
    main()
