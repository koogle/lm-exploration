"""Linking per-frame detections into tracks, for video.

`Tracker.update(boxes, scores, labels)` takes one frame's detections and returns the
frame's tracked boxes. Each detection joins the existing track of its label it
overlaps most (Hungarian matching on IoU, at least `match_iou`), or starts a new
track. A track's box follows its detections with exponential smoothing (`smoothing`
is the weight of the previous box). A track that finds no detection is still reported
at its last box for up to `coast` frames, then dropped. Reported tracks carry a stable
id, so they can be scored as tracks as well as per frame.
"""

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

from parallel_detection.boxes import iou_matrix


class Tracker:
    def __init__(self, match_iou=0.3, smoothing=0.5, coast=1):
        self.match_iou, self.smoothing, self.coast = match_iou, smoothing, coast
        self.tracks = []  # {"id", "label", "box", "score", "missed"}
        self.ids = itertools.count()

    def update(self, boxes, scores, labels):
        """Tracked boxes of this frame: [{"id", "label", "box", "score", "coasting"}]."""
        matched_tracks, matched_found = set(), set()
        for label in set(labels) | {t["label"] for t in self.tracks}:
            tracks = [i for i, t in enumerate(self.tracks) if t["label"] == label]
            found = [i for i, x in enumerate(labels) if x == label]
            if not tracks or not found:
                continue
            iou = iou_matrix([self.tracks[i]["box"] for i in tracks], [boxes[j] for j in found])
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols, strict=True):
                if iou[r, c] < self.match_iou:
                    continue
                track, j = self.tracks[tracks[r]], found[c]
                track["box"] = (
                    self.smoothing * np.asarray(track["box"])
                    + (1 - self.smoothing) * np.asarray(boxes[j])
                ).tolist()
                track["score"], track["missed"] = scores[j], 0
                matched_tracks.add(tracks[r])
                matched_found.add(j)
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track["missed"] += 1
        self.tracks = [t for t in self.tracks if t["missed"] <= self.coast]
        for j, (box, score, label) in enumerate(zip(boxes, scores, labels, strict=True)):
            if j not in matched_found:
                self.tracks.append(
                    {
                        "id": next(self.ids),
                        "label": label,
                        "box": list(box),
                        "score": score,
                        "missed": 0,
                    }
                )
        return [
            {
                "id": t["id"],
                "label": t["label"],
                "box": t["box"],
                "score": t["score"],
                "coasting": t["missed"] > 0,
            }
            for t in self.tracks
        ]
