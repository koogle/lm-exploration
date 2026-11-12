"""Evaluation tasks in one layout: an image and the phrases to detect in it.

A sample is {"dataset", "image_id", "image_url", "width", "height", "queries"}.
A query is {"phrase", "kind", "boxes", "exhaustive", "group"}:

- kind "present": every instance of a category that is in the image.
- kind "absent": a category verified not to be in the image; any box is a false alarm.
- kind "referring": an expression that picks out exactly one object.

`exhaustive` is False when the dataset does not promise that every instance of the
category is annotated, so unmatched predictions for that query are not errors.
"""

import json
import random
import re
from collections import defaultdict

COCO_IMAGES = "http://images.cocodataset.org"


def lvis_phrase(name):
    """LVIS names look like `bow_(weapon)`; prompts read as `bow (weapon)`."""
    return name.replace("_", " ")


def lvis_samples(annotations, images, max_absent, seed):
    """Prompt each image with its annotated categories and verified-absent ones.

    LVIS is federated: an image lists the categories checked and found absent
    (`neg_category_ids`) and those present but not fully annotated
    (`not_exhaustive_category_ids`).
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
                "dataset": "lvis",
                "image_id": image["id"],
                "image_url": image["coco_url"],
                "width": image["width"],
                "height": image["height"],
                "queries": queries,
            }
        )
    return samples


def refcoco_samples(rows, refs, seed):
    """One seeded expression per referred object, grouped by image.

    `rows` follow the jxu124/refcoco* layout: one row per referred object with its
    expressions, the COCO annotation and image record as JSON strings.
    """
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda row: row["ref_id"])
    by_image = defaultdict(list)
    for row in rng.sample(rows, min(refs, len(rows))):
        by_image[row["image_id"]].append(row)
    samples = []
    for image_id, group in sorted(by_image.items()):
        info = json.loads(group[0]["raw_image_info"])
        queries = []
        for row in group:
            x, y, w, h = json.loads(row["raw_anns"])["bbox"]
            queries.append(
                {
                    "phrase": rng.choice(row["sentences"])["raw"].strip(),
                    "kind": "referring",
                    "boxes": [[x, y, x + w, y + h]],
                    "exhaustive": True,
                    "group": "",
                }
            )
        folder = re.fullmatch(r"COCO_(\w+2014)_\d+\.jpg", info["file_name"]).group(1)
        samples.append(
            {
                "dataset": "refcoco",
                "image_id": image_id,
                "image_url": f"{COCO_IMAGES}/{folder}/{info['file_name']}",
                "width": info["width"],
                "height": info["height"],
                "queries": queries,
            }
        )
    return samples
