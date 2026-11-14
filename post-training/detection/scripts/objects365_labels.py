"""Human Objects365 boxes for prepared Objects365 images, in the teacher-label format.

    python scripts/objects365_labels.py --manifest artifacts/teacher-images/manifest.jsonl \
        --same-images artifacts/teacher-train/google--gemini-4-carbon.jsonl \
        --out artifacts/objects365-human/objects365-human.jsonl

Reads Objects365 v1 training annotations (`jxu124/objects365`) and writes one JSON line
per prepared Objects365 image, like `teacher.label` output: image id, the stored image's
size, and objects with a COCO category name and a box in pixels of the stored image.
Objects365 categories map to COCO's 80 (`TO_COCO`); others are dropped, as are crowd
boxes. `--same-images` keeps only the images that labels file holds, so a run can swap
one label source for another on the same images.
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from parallel_detection.tasks import COCO_NAMES

REPO, REVISION = "jxu124/objects365", "b9ce8d3596a3642fb909226750017d75dbc669a5"
SHARDS = [
    "data/train-00000-of-00007-84574fae7b700510.parquet",
    "data/train-00001-of-00007-cce9fefd2db5c6e2.parquet",
    "data/train-00002-of-00007-1f1c8fbd070462de.parquet",
    "data/train-00003-of-00007-2adf7bbbadb95884.parquet",
    "data/train-00004-of-00007-d8aaf60c95e4d979.parquet",
    "data/train-00005-of-00007-da3b6be679c1be5a.parquet",
    "data/train-00006-of-00007-ae825883aca8a8fb.parquet",
]
# COCO category -> the Objects365 categories it covers (names as the annotations spell them).
FROM_COCO = {
    "person": ["Person"],
    "bicycle": ["Bicycle"],
    "car": ["Car", "SUV", "Sports Car", "Van"],
    "motorcycle": ["Motorcycle"],
    "airplane": ["Airplane"],
    "bus": ["Bus"],
    "train": ["Train"],
    "truck": ["Truck", "Pickup Truck", "Heavy Truck", "Fire Truck"],
    "boat": ["Boat", "Sailboat", "Ship"],
    "traffic light": ["Traffic Light"],
    "fire hydrant": ["Fire Hydrant"],
    "stop sign": ["Stop Sign"],
    "parking meter": ["Parking meter"],
    "bench": ["Bench"],
    "bird": ["Wild Bird", "Pigeon", "Duck", "Goose", "Swan", "Parrot", "Chicken", "Penguin"],
    "cat": ["Cat"],
    "dog": ["Dog"],
    "horse": ["Horse"],
    "sheep": ["Sheep"],
    "cow": ["Cow"],
    "elephant": ["Elephant"],
    "bear": ["Bear"],
    "zebra": ["Zebra"],
    "giraffe": ["Giraffe"],
    "backpack": ["Backpack"],
    "umbrella": ["Umbrella"],
    "handbag": ["Handbag/Satchel"],
    "tie": ["Tie", "Bow Tie"],
    "suitcase": ["Luggage"],
    "frisbee": ["Frisbee"],
    "skis": ["Skiboard"],
    "snowboard": ["Snowboard"],
    "sports ball": [
        "Baseball",
        "Basketball",
        "Soccer",
        "Volleyball",
        "American Football",
        "Golf Ball",
        "Tennis",
        "Table Tennis ",
        "Other Balls",
    ],
    "kite": ["Kite"],
    "baseball bat": ["Baseball Bat"],
    "baseball glove": ["Baseball Glove"],
    "skateboard": ["Skateboard"],
    "surfboard": ["Surfboard"],
    "tennis racket": ["Tennis Racket"],
    "bottle": ["Bottle"],
    "wine glass": ["Wine Glass"],
    "cup": ["Cup"],
    "fork": ["Fork"],
    "knife": ["Knife"],
    "spoon": ["Spoon"],
    "bowl": ["Bowl/Basin"],
    "banana": ["Banana"],
    "apple": ["Apple"],
    "sandwich": ["Sandwich"],
    "orange": ["Orange/Tangerine"],
    "broccoli": ["Broccoli"],
    "carrot": ["Carrot"],
    "hot dog": ["Hot dog"],
    "pizza": ["Pizza"],
    "donut": ["Donut"],
    "cake": ["Cake"],
    "chair": ["Chair"],
    "couch": ["Couch"],
    "potted plant": ["Potted Plant"],
    "bed": ["Bed"],
    "dining table": ["Dinning Table"],
    "toilet": ["Toilet"],
    "tv": ["Moniter/TV"],
    "laptop": ["Laptop"],
    "mouse": ["Mouse"],
    "remote": ["Remote"],
    "keyboard": ["Keyboard"],
    "cell phone": ["Cell Phone"],
    "microwave": ["Microwave"],
    "oven": ["Oven"],
    "toaster": ["Toaster"],
    "sink": ["Sink"],
    "refrigerator": ["Refrigerator"],
    "book": ["Book"],
    "clock": ["Clock"],
    "vase": ["Vase"],
    "scissors": ["Scissors"],
    "teddy bear": ["Stuffed Toy"],
    "hair drier": ["Hair Dryer"],
    "toothbrush": ["Toothbrush"],
}
TO_COCO = {o365: coco for coco, names in FROM_COCO.items() for o365 in names}
assert sorted(FROM_COCO) == sorted(COCO_NAMES)


def labels(manifest, keep=None):
    """Teacher-format replies for the manifest's Objects365 images (those in `keep`)."""
    stored = {
        Path(r["image"]).name: r
        for r in map(json.loads, Path(manifest).read_text().splitlines())
        if r["source"] == "objects365" and (keep is None or r["image_id"] in keep)
    }
    for shard in SHARDS:
        path = hf_hub_download(REPO, shard, repo_type="dataset", revision=REVISION)
        table = pq.read_table(path, columns=["image_path", "image_info", "anns_info"])
        for row in table.to_pylist():
            image = stored.get(Path(row["image_path"]).name)
            if image is None:
                continue
            sx = image["width"] / row["image_info"]["width"]
            sy = image["height"] / row["image_info"]["height"]
            objects = [
                {
                    "label": TO_COCO[a["category"]],
                    "box": [
                        a["bbox"][0] * sx,
                        a["bbox"][1] * sy,
                        a["bbox"][2] * sx,
                        a["bbox"][3] * sy,
                    ],
                }
                for a in row["anns_info"]
                if a["category"] in TO_COCO and not a["iscrowd"]
            ]
            yield {
                "model": "objects365-human",
                "image_id": image["image_id"],
                "width": image["width"],
                "height": image["height"],
                "objects": objects,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--same-images", help="a labels file whose images to keep")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    keep = None
    if args.same_images:
        replies = map(json.loads, Path(args.same_images).read_text().splitlines())
        keep = {r["image_id"] for r in replies if "error" not in r}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(labels(args.manifest, keep))
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"images": len(rows), "objects": sum(len(r["objects"]) for r in rows)}))


if __name__ == "__main__":
    main()
