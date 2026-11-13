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
