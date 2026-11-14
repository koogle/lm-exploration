"""Evaluation tasks in one layout: an image and the phrases to detect in it.

A sample is {"image_id", "image", "width", "height", "queries"}. "image" is a URL or
a path relative to the image folder. A query is {"phrase", "kind", "boxes",
"exhaustive", "group"}:

- kind "present": every instance of a category that is in the image.
- kind "referring": an expression that picks out one object.
- kind "absent": a category verified not to be in the image; any box is a false alarm.

`exhaustive` is False when the dataset does not promise that every instance of the
category is annotated, so unmatched predictions for that query are not errors.
"""

import random
from collections import defaultdict
from pathlib import Path

# COCO's 80 detection categories, in the class-index order of detection-datasets/coco.
COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def lvis_phrase(name):
    """LVIS names look like `bow_(weapon)`; prompts read as `bow (weapon)`."""
    return name.replace("_", " ")


def lvis_samples(annotations, images, max_absent, seed):
    """Prompt each image with its annotated categories.

    LVIS is federated: an image lists the categories checked and found absent
    (`neg_category_ids`) and those present but not fully annotated
    (`not_exhaustive_category_ids`). `max_absent` adds that many absent categories.
    """
    rng = random.Random(seed)
    categories = {c["id"]: c for c in annotations["categories"]}
    by_image = defaultdict(lambda: defaultdict(list))
    for ann in annotations["annotations"]:
        x, y, w, h = ann["bbox"]
        by_image[ann["image_id"]][ann["category_id"]].append([x, y, x + w, y + h])
    chosen = sorted(annotations["images"], key=lambda image: image["id"])
    chosen = [image for image in chosen if by_image[image["id"]]]
    samples = []
    for image in rng.sample(chosen, min(images, len(chosen))):
        partial = set(image["not_exhaustive_category_ids"])
        queries = [
            {
                "phrase": lvis_phrase(categories[cid]["name"]),
                "kind": "present",
                "boxes": boxes,
                "exhaustive": cid not in partial,
                "group": categories[cid]["frequency"],
            }
            for cid, boxes in sorted(by_image[image["id"]].items())
        ]
        absent = sorted(image["neg_category_ids"])
        for cid in sorted(rng.sample(absent, min(max_absent, len(absent)))):
            queries.append(
                {
                    "phrase": lvis_phrase(categories[cid]["name"]),
                    "kind": "absent",
                    "boxes": [],
                    "exhaustive": True,
                    "group": categories[cid]["frequency"],
                }
            )
        samples.append(
            {
                "image_id": image["id"],
                "image": image["coco_url"],
                "width": image["width"],
                "height": image["height"],
                "queries": queries,
            }
        )
    return samples


def coco_samples(rows, images, seed, names=COCO_NAMES, exclude_images=0):
    """Prompt each COCO image with the categories annotated in it.

    `rows` follow detection-datasets/coco: `objects` holds parallel `category`
    indices into `names` and `bbox` as [x0, y0, x1, y1]. `exclude_images` leaves out
    the images that a sample of that size with the same seed would take, so a tuning
    split never overlaps the evaluation split.
    """
    rows = [row for row in sorted(rows, key=lambda row: row["image_id"]) if row["objects"]["bbox"]]
    if exclude_images:
        taken = {s["image_id"] for s in coco_samples(rows, exclude_images, seed, names)}
        rows = [row for row in rows if row["image_id"] not in taken]
    rng = random.Random(seed)
    samples = []
    for row in rng.sample(rows, min(images, len(rows))):
        by_category = defaultdict(list)
        for category, box in zip(row["objects"]["category"], row["objects"]["bbox"], strict=True):
            by_category[category].append(list(box))
        samples.append(
            {
                "image_id": row["image_id"],
                "image": f"coco/{row['image_id']}.jpg",
                "width": row["width"],
                "height": row["height"],
                "queries": [
                    {
                        "phrase": names[category],
                        "kind": "present",
                        "boxes": boxes,
                        "exhaustive": True,
                        "group": "",
                    }
                    for category, boxes in sorted(by_category.items())
                ],
            }
        )
    return samples


# Pascal VOC's 20 categories under their COCO names, so every model is prompted the same
# way on both datasets.
VOC_TO_COCO = {
    "aeroplane": "airplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "diningtable": "dining table",
    "dog": "dog",
    "horse": "horse",
    "motorbike": "motorcycle",
    "person": "person",
    "pottedplant": "potted plant",
    "sheep": "sheep",
    "sofa": "couch",
    "train": "train",
    "tvmonitor": "tv",
}


def voc_annotation(xml):
    """One Pascal VOC annotation file as {image_id, width, height, objects}.

    Boxes become [x0, y0, x1, y1] with VOC's 1-based pixel corners moved to 0-based;
    objects keep VOC's `difficult` flag.
    """
    from xml.etree import ElementTree

    root = ElementTree.fromstring(xml)
    size = root.find("size")
    objects = []
    for obj in root.iter("object"):
        box = obj.find("bndbox")
        corners = [float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        objects.append(
            {
                "name": VOC_TO_COCO[obj.find("name").text.strip()],
                "box": [corners[0] - 1, corners[1] - 1, corners[2], corners[3]],
                "difficult": obj.findtext("difficult", "0").strip() == "1",
            }
        )
    return {
        "image_id": Path(root.findtext("filename")).stem,
        "width": int(size.findtext("width")),
        "height": int(size.findtext("height")),
        "objects": objects,
    }


def voc_samples(annotations, images, seed):
    """Prompt each sampled VOC image with the categories annotated in it.

    VOC labels every instance of its 20 categories, so queries are exhaustive. Objects
    VOC marks `difficult` count as truth here like any other; the "group" of a query
    says whether it holds one.
    """
    rows = sorted((a for a in annotations if a["objects"]), key=lambda a: a["image_id"])
    samples = []
    for row in random.Random(seed).sample(rows, min(images, len(rows))):
        by_category = defaultdict(list)
        for obj in row["objects"]:
            by_category[obj["name"]].append(obj)
        samples.append(
            {
                "image_id": row["image_id"],
                "image": f"voc2007/{row['image_id']}.jpg",
                "width": row["width"],
                "height": row["height"],
                "queries": [
                    {
                        "phrase": name,
                        "kind": "present",
                        "boxes": [o["box"] for o in objs],
                        "exhaustive": True,
                        "group": "difficult" if any(o["difficult"] for o in objs) else "",
                    }
                    for name, objs in sorted(by_category.items())
                ],
            }
        )
    return samples


# MOT17 ground-truth classes that are people: pedestrian, person on a vehicle and static
# person. MOT's own scoring ignores the last two; here they are people to find.
MOT_PERSON_CLASSES = {1, 2, 7}


def mot_samples(sequence, ground_truth, width, height, every):
    """Every `every`-th frame of a MOT17 sequence, prompted with "person".

    `ground_truth` is the sequence's gt.txt (frame, id, x, y, w, h, flag, class,
    visibility per line); boxes become [x0, y0, x1, y1] in pixels.
    """
    boxes = defaultdict(list)
    for line in ground_truth.splitlines():
        frame, _, x, y, w, h, _, kind, _ = line.split(",")
        if int(kind) in MOT_PERSON_CLASSES:
            x0, y0 = float(x) - 1, float(y) - 1
            boxes[int(frame)].append([x0, y0, x0 + float(w), y0 + float(h)])
    return [
        {
            "image_id": f"{sequence}/{frame:06d}",
            "image": f"mot17/{sequence}/{frame:06d}.jpg",
            "width": width,
            "height": height,
            "queries": [
                {
                    "phrase": "person",
                    "kind": "present",
                    "boxes": boxes[frame],
                    "exhaustive": True,
                    "group": sequence,
                }
            ],
        }
        for frame in sorted(boxes)
        if (frame - 1) % every == 0
    ]


def refcoco_m_samples(rows, images, seed):
    """Every expression of every referred object in the sampled images.

    `rows` follow moondream/refcoco-m: one row per image, `samples` holding each
    referred object's `sentences`, `bbox` as [x, y, width, height] and `mask` size.
    """
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda row: row["image_id"])
    samples = []
    for row in rng.sample(rows, min(images, len(rows))):
        height, width = row["samples"][0]["mask"]["size"]
        queries = []
        for referred in row["samples"]:
            x, y, w, h = referred["bbox"]
            for sentence in referred["sentences"]:
                queries.append(
                    {
                        "phrase": sentence.strip(),
                        "kind": "referring",
                        "boxes": [[x, y, x + w, y + h]],
                        "exhaustive": True,
                        "group": "",
                    }
                )
        samples.append(
            {
                "image_id": row["image_id"],
                "image": f"refcoco-m/{row['image_id']}.jpg",
                "width": width,
                "height": height,
                "queries": queries,
            }
        )
    return samples
