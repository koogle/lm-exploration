"""Box geometry and one-to-one matching. Boxes are [x0, y0, x1, y1] in pixels."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def as_boxes(boxes):
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def areas(boxes):
    boxes = as_boxes(boxes)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def iou_matrix(a, b):
    """Pairwise intersection over union, shape (len(a), len(b))."""
    a, b = as_boxes(a), as_boxes(b)
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(high - low, 0, None).prod(axis=2)
    union = areas(a)[:, None] + areas(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match(predicted, truth, threshold):
    """Largest one-to-one matching of predictions to truth boxes with IoU >= threshold.

    The predictions carry no confidence, so no greedy order exists; this takes the
    matching with the most pairs and, among those, the highest total IoU. Returns
    (prediction index, truth index) pairs.
    """
    iou = iou_matrix(predicted, truth)
    if iou.size == 0:
        return []
    valid = iou >= threshold
    # An allowed pair is always worth more than any total-IoU gain (bounded by the pair count).
    rows, cols = linear_sum_assignment(
        np.where(valid, min(iou.shape) + 1 + iou, 0.0), maximize=True
    )
    return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if valid[r, c]]
