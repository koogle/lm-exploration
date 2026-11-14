"""How many objects, and of what size, each teacher labelled per image, by image source.

    python scripts/label_stats.py artifacts/teacher-train/google--gemini-4-carbon.jsonl \
        artifacts/teacher-train-argon/google--gemini-4-argon.jsonl > artifacts/label-stats.json

Reads labelling replies (`teacher.label` output) and writes JSON: per file and source,
the images, the share with no object, and objects per image in all and by box area as a
share of the image (under 0.3%, 0.3% to 1%, 1% and over).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

BUCKETS = (("under 0.3%", 0.0, 0.003), ("0.3% to 1%", 0.003, 0.01), ("1% and over", 0.01, 2.0))


def stats(replies):
    by_source = defaultdict(list)
    for reply in replies:
        by_source[reply["image_id"].split("/")[0]].append(reply)
    out = {}
    for source, rows in sorted(by_source.items()):
        counts = dict.fromkeys(name for name, _, _ in BUCKETS)
        for name, low, high in BUCKETS:
            counts[name] = sum(
                low
                <= (o["box"][2] - o["box"][0])
                * (o["box"][3] - o["box"][1])
                / (r["width"] * r["height"])
                < high
                for r in rows
                for o in r["objects"]
            )
        out[source] = {
            "images": len(rows),
            "share_without_objects": sum(not r["objects"] for r in rows) / len(rows),
            "objects_per_image": sum(len(r["objects"]) for r in rows) / len(rows),
            "objects_per_image_by_area": {name: n / len(rows) for name, n in counts.items()},
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", nargs="+", type=Path)
    args = parser.parse_args()
    result = {}
    for path in args.labels:
        replies = [json.loads(line) for line in path.read_text().splitlines()]
        replies = [r for r in replies if "error" not in r]
        result[path.name] = {"model": replies[0]["model"]} | stats(replies)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
