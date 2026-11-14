"""Choosing which slots to report: one view or several, labels, duplicates.

A candidate is a dict with "box" ([x0, y0, x1, y1] in pixels of the full image),
"label", "score", and for tiled inference "view" ("full" or "tile") and "cut" (the box
touches a tile edge that is not an image edge, so the object may be cut off).

`select` turns candidates into per-phrase boxes and scores. The steps, in order:

1. Views: full-image candidates always count; a tile candidate counts when it is not
   cut and its box is smaller than `tile_max_cells` grid cells (the side of a square of
   the same area, in cells of the 27 x 27 image grid of the full image).
2. Labels: a phrase takes candidates whose label is one of `aliases.get(phrase,
   [phrase])`, so a dataset's category can cover several head labels.
3. Threshold on score. On video, `previous` (the last frame's boxes per phrase) lets a
   candidate through from `keep_threshold` when it overlaps a previous box of its
   phrase by IoU 0.5 (hysteresis: objects already found need less evidence to stay).
4. Duplicates, among the phrase's candidates: "nms" keeps the best-scoring box and drops
   boxes overlapping a kept one by at least `duplicate_iou`; "fuse" (weighted box
   fusion) replaces each such group by its score-weighted mean box with the group's
   best score.
"""

import numpy as np

from parallel_detection.boxes import iou_matrix

GRID = 27


def cells(box, width, height):
    """Side of a square of the box's area, in cells of the full image's grid."""
    w = max(box[2] - box[0], 0.0) * GRID / width
    h = max(box[3] - box[1], 0.0) * GRID / height
    return float(np.sqrt(w * h))


def usable(candidate, width, height, tile_max_cells):
    if candidate.get("view", "full") == "full":
        return True
    if tile_max_cells is None or candidate.get("cut"):
        return False
    return cells(candidate["box"], width, height) < tile_max_cells


def nms(candidates, duplicate_iou):
    kept = []
    for c in sorted(candidates, key=lambda c: -c["score"]):
        if (
            kept
            and duplicate_iou is not None
            and iou_matrix([c["box"]], [k["box"] for k in kept]).max() >= duplicate_iou
        ):
            continue
        kept.append(c)
    return kept


def fuse(candidates, duplicate_iou):
    groups = []  # [fused box, best score, members]
    for c in sorted(candidates, key=lambda c: -c["score"]):
        if groups:
            overlap = iou_matrix([c["box"]], [g[0] for g in groups])[0]
            best = int(overlap.argmax())
            if overlap[best] >= duplicate_iou:
                members = groups[best][2] + [c]
                weights = np.array([m["score"] for m in members])
                boxes = np.array([m["box"] for m in members], dtype=np.float64)
                groups[best] = [
                    (weights[:, None] * boxes).sum(0) / weights.sum(),
                    groups[best][1],
                    members,
                ]
                continue
        groups.append([np.asarray(c["box"], dtype=np.float64), c["score"], [c]])
    return [{"box": g[0].tolist(), "score": g[1]} for g in groups]


MERGES = {"nms": nms, "fuse": fuse}


def tile_grid(width, height, tiles, overlap):
    """[x0, y0, x1, y1] of a `tiles` x `tiles` grid of crops covering the image, each
    overlapping its neighbours by `overlap` of a tile."""
    tile_w = width / (1 + (tiles - 1) * (1 - overlap))
    tile_h = height / (1 + (tiles - 1) * (1 - overlap))
    return [
        [x0, y0, x0 + tile_w, y0 + tile_h]
        for row in range(tiles)
        for col in range(tiles)
        for x0, y0 in [(col * tile_w * (1 - overlap), row * tile_h * (1 - overlap))]
    ]


def add_unmatched(objects, extra, iou):
    """`objects` ({"label", "box"}) plus each of `extra` that no object of its label
    already overlaps by `iou`; later extras see the ones added before them."""
    out = list(objects)
    for o in extra:
        same = [x["box"] for x in out if x["label"] == o["label"]]
        if not same or iou_matrix([o["box"]], same).max() < iou:
            out.append(o)
    return out


def select(
    candidates,
    phrases,
    width,
    height,
    threshold,
    duplicate_iou=0.7,
    merge="nms",
    aliases=None,
    tile_max_cells=None,
    previous=None,
    keep_threshold=None,
):
    """Per phrase, the reported boxes and scores in descending score."""
    if merge not in MERGES:
        raise ValueError(f"Unknown merge {merge!r}")
    aliases = aliases or {}
    floor = threshold if previous is None or keep_threshold is None else keep_threshold
    pool = [
        c
        for c in candidates
        if c["label"] is not None
        and c["score"] >= floor
        and usable(c, width, height, tile_max_cells)
    ]
    boxes, scores = [], []
    for phrase in phrases:
        labels = set(aliases.get(phrase, [phrase]))
        before = (previous or {}).get(phrase, [])
        mine = [
            c
            for c in pool
            if c["label"] in labels
            and (
                c["score"] >= threshold or (before and iou_matrix([c["box"]], before).max() >= 0.5)
            )
        ]
        kept = sorted(MERGES[merge](mine, duplicate_iou), key=lambda c: -c["score"])
        boxes.append([list(c["box"]) for c in kept])
        scores.append([c["score"] for c in kept])
    return boxes, scores
