"""Human boxes for prepared Objects365, Open Images or COCO images, in the teacher-label
format.

    python scripts/human_labels.py --source objects365 \
        --manifest artifacts/teacher-images/manifest.jsonl \
        --same-images artifacts/teacher-train/google--gemini-4-carbon.jsonl \
        --out artifacts/human-labels/objects365-human.jsonl

Writes one JSON line per prepared image of the source, like `teacher.label` output:
image id, the stored image's size, and objects with a COCO category name and a box in
pixels of the stored image. Each source's categories map to COCO's 80 (`FROM_COCO`);
others are dropped, as are crowd and group-of boxes. `--same-images` keeps only the
images that labels file holds, so a run can swap one label source for another on the
same images. The most frequent unmapped categories are printed for checking the map.

- objects365: Objects365 v1 training annotations (`jxu124/objects365`), joined by file
  name.
- openimages: the human-verified boxes stored with the images in
  `vikhyatk/openimages-bbox`, joined by shard and row. Open Images boxes every instance
  of the classes verified present in an image, so classes never verified can be
  missing.
- coco: COCO's own boxes from the `detection-datasets/coco` Parquet shards that
  `parallel_detection.images` prepared the images from (the `--images-config` source).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from parallel_detection.tasks import COCO_NAMES

O365_REPO, O365_REVISION = "jxu124/objects365", "b9ce8d3596a3642fb909226750017d75dbc669a5"
OPENIMAGES_REPO = "vikhyatk/openimages-bbox"
OPENIMAGES_REVISION, OPENIMAGES_SHARDS = "9c9a38be0f64b5f6caf2fcac4a00e2b155f1b7ad", 1103
O365_SHARDS = [
    "data/train-00000-of-00007-84574fae7b700510.parquet",
    "data/train-00001-of-00007-cce9fefd2db5c6e2.parquet",
    "data/train-00002-of-00007-1f1c8fbd070462de.parquet",
    "data/train-00003-of-00007-2adf7bbbadb95884.parquet",
    "data/train-00004-of-00007-d8aaf60c95e4d979.parquet",
    "data/train-00005-of-00007-da3b6be679c1be5a.parquet",
    "data/train-00006-of-00007-ae825883aca8a8fb.parquet",
]
# COCO category -> the Objects365 categories it covers (names as the annotations spell them).
O365_FROM_COCO = {
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
BIRDS = ["Bird", "Duck", "Goose", "Swan", "Eagle", "Owl", "Parrot", "Penguin", "Chicken"]
BIRDS += ["Sparrow", "Woodpecker", "Blue jay", "Magpie", "Raven", "Ostrich", "Falcon"]
BIRDS += ["Turkey", "Canary"]
BALLS = ["Ball", "Football", "Tennis ball", "Golf ball", "Rugby ball", "Cricket ball"]
BALLS += ["Volleyball (Ball)"]
# COCO category -> the Open Images classes it covers.
OPENIMAGES_FROM_COCO = {
    "person": ["Person", "Man", "Woman", "Boy", "Girl"],
    "bicycle": ["Bicycle"],
    "car": ["Car", "Taxi", "Van", "Limousine"],
    "motorcycle": ["Motorcycle"],
    "airplane": ["Airplane"],
    "bus": ["Bus"],
    "train": ["Train"],
    "truck": ["Truck"],
    "boat": ["Boat", "Canoe", "Gondola", "Barge"],
    "traffic light": ["Traffic light"],
    "fire hydrant": ["Fire hydrant"],
    "stop sign": ["Stop sign"],
    "parking meter": ["Parking meter"],
    "bench": ["Bench"],
    "bird": BIRDS,
    "cat": ["Cat"],
    "dog": ["Dog"],
    "horse": ["Horse"],
    "sheep": ["Sheep"],
    "cow": ["Cattle", "Bull"],
    "elephant": ["Elephant"],
    "bear": ["Bear", "Brown bear", "Polar bear"],
    "zebra": ["Zebra"],
    "giraffe": ["Giraffe"],
    "backpack": ["Backpack"],
    "umbrella": ["Umbrella"],
    "handbag": ["Handbag"],
    "tie": ["Tie"],
    "suitcase": ["Suitcase"],
    "frisbee": ["Flying disc"],
    "skis": ["Ski"],
    "snowboard": ["Snowboard"],
    "sports ball": BALLS,
    "kite": ["Kite"],
    "baseball bat": ["Baseball bat"],
    "baseball glove": ["Baseball glove"],
    "skateboard": ["Skateboard"],
    "surfboard": ["Surfboard"],
    "tennis racket": ["Tennis racket"],
    "bottle": ["Bottle"],
    "wine glass": ["Wine glass"],
    "cup": ["Coffee cup", "Mug"],
    "fork": ["Fork"],
    "knife": ["Knife", "Kitchen knife"],
    "spoon": ["Spoon"],
    "bowl": ["Bowl", "Mixing bowl"],
    "banana": ["Banana"],
    "apple": ["Apple"],
    "sandwich": ["Sandwich", "Submarine sandwich"],
    "orange": ["Orange"],
    "broccoli": ["Broccoli"],
    "carrot": ["Carrot"],
    "hot dog": ["Hot dog"],
    "pizza": ["Pizza"],
    "donut": ["Doughnut"],
    "cake": ["Cake"],
    "chair": ["Chair"],
    "couch": ["Couch", "Sofa bed", "Studio couch", "Loveseat"],
    "potted plant": ["Houseplant"],
    "bed": ["Bed"],
    "dining table": ["Kitchen & dining room table", "Table"],
    "toilet": ["Toilet"],
    "tv": ["Television", "Computer monitor"],
    "laptop": ["Laptop"],
    "mouse": ["Computer mouse"],
    "remote": ["Remote control"],
    "keyboard": ["Computer keyboard"],
    "cell phone": ["Mobile phone"],
    "microwave": ["Microwave oven"],
    "oven": ["Oven"],
    "toaster": ["Toaster"],
    "sink": ["Sink"],
    "refrigerator": ["Refrigerator"],
    "book": ["Book"],
    "clock": ["Clock", "Alarm clock", "Wall clock", "Digital clock"],
    "vase": ["Vase"],
    "scissors": ["Scissors"],
    "teddy bear": ["Teddy bear"],
    "hair drier": ["Hair dryer"],
    "toothbrush": ["Toothbrush"],
}
FROM_COCO = {
    "objects365": O365_FROM_COCO,
    "openimages": OPENIMAGES_FROM_COCO,
    "coco": {name: [name] for name in COCO_NAMES},
}
assert all(sorted(m) == sorted(COCO_NAMES) for m in FROM_COCO.values())


def to_coco(source):
    return {name: coco for coco, names in FROM_COCO[source].items() for name in names}


def stored(manifest, source, keep):
    return {
        r["image_id"]: r
        for r in map(json.loads, Path(manifest).read_text().splitlines())
        if r["source"] == source and (keep is None or r["image_id"] in keep)
    }


def reply(image, objects):
    return {
        "model": f"{image['source']}-human",
        "image_id": image["image_id"],
        "width": image["width"],
        "height": image["height"],
        "objects": objects,
    }


def objects365(images, unmapped):
    mapping = to_coco("objects365")
    by_name = {Path(r["image"]).name: r for r in images.values()}
    for shard in O365_SHARDS:
        path = hf_hub_download(O365_REPO, shard, repo_type="dataset", revision=O365_REVISION)
        table = pq.read_table(path, columns=["image_path", "image_info", "anns_info"])
        for row in table.to_pylist():
            image = by_name.get(Path(row["image_path"]).name)
            if image is None:
                continue
            sx = image["width"] / row["image_info"]["width"]
            sy = image["height"] / row["image_info"]["height"]
            objects = []
            for a in row["anns_info"]:
                if a["iscrowd"]:
                    continue
                if a["category"] not in mapping:
                    unmapped[a["category"]] += 1
                    continue
                x0, y0, x1, y1 = a["bbox"]
                objects.append(
                    {"label": mapping[a["category"]], "box": [x0 * sx, y0 * sy, x1 * sx, y1 * sy]}
                )
            yield reply(image, objects)


def openimages(images, unmapped):
    mapping = to_coco("openimages")
    by_shard = {}
    for image_id, image in images.items():
        shard, row = image_id.split("/")[1].split("-")
        by_shard.setdefault(int(shard), {})[int(row)] = image
    for shard, rows in sorted(by_shard.items()):
        name = f"data/train-{shard:05d}-of-{OPENIMAGES_SHARDS:05d}.parquet"
        path = Path(
            hf_hub_download(
                OPENIMAGES_REPO, name, repo_type="dataset", revision=OPENIMAGES_REVISION
            )
        )
        column = pq.read_table(path, columns=["objects"]).column("objects")
        for row, image in sorted(rows.items()):
            objects = []
            for o in column[row].as_py():
                if o["is_group_of"]:
                    continue
                if o["label"] not in mapping:
                    unmapped[o["label"]] += 1
                    continue
                w, h = image["width"], image["height"]
                box = [o["xmin"] * w, o["ymin"] * h, o["xmax"] * w, o["ymax"] * h]
                objects.append({"label": mapping[o["label"]], "box": box})
            yield reply(image, objects)
        # Shards hold the images too; keep one on disk at a time.
        path.resolve().unlink()
        path.unlink()


def coco(images, unmapped, settings):
    """COCO's boxes (category indices into COCO_NAMES, [x0, y0, x1, y1] pixels), scaled
    to the stored image."""
    by_id = {image_id.split("/")[1]: image for image_id, image in images.items()}
    for name in settings["files"]:
        path = hf_hub_download(
            settings["repo"], name, repo_type="dataset", revision=settings["revision"]
        )
        table = pq.read_table(path, columns=["image_id", "width", "height", "objects"])
        for row in table.to_pylist():
            image = by_id.get(str(row["image_id"]))
            if image is None:
                continue
            sx, sy = image["width"] / row["width"], image["height"] / row["height"]
            objects = [
                {"label": COCO_NAMES[c], "box": [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]}
                for c, b in zip(row["objects"]["category"], row["objects"]["bbox"], strict=True)
            ]
            yield reply(image, objects)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=FROM_COCO, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--same-images", help="a labels file whose images to keep")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images-config", help="the images config with the coco source")
    args = parser.parse_args()
    keep = None
    if args.same_images:
        replies = map(json.loads, Path(args.same_images).read_text().splitlines())
        keep = {r["image_id"] for r in replies if "error" not in r}
    images = stored(args.manifest, args.source, keep)
    unmapped = Counter()
    if args.source == "coco":
        settings = json.loads(Path(args.images_config).read_text())["sources"]["coco"]
        rows = list(coco(images, unmapped, settings))
    else:
        rows = list(
            {"objects365": objects365, "openimages": openimages}[args.source](images, unmapped)
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(
        json.dumps(
            {
                "images": len(rows),
                "objects": sum(len(r["objects"]) for r in rows),
                "most_frequent_unmapped": unmapped.most_common(40),
            }
        )
    )


if __name__ == "__main__":
    main()
