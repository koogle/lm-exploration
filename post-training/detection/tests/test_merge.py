from parallel_detection.merge import cells, select


def candidate(box, label, score, view="full", cut=False):
    return {"box": box, "label": label, "score": score, "view": view, "cut": cut}


def test_select_thresholds_ranks_and_drops_same_label_duplicates():
    found = [
        candidate([0, 0, 10, 10], "cup", 0.5),
        candidate([0, 0, 10, 9], "cup", 0.9),
        candidate([0, 0, 10, 10], "dog", 0.8),
        candidate([50, 50, 60, 60], "cup", 0.2),
        candidate([0, 0, 1, 1], None, 0.99),
    ]
    boxes, scores = select(found, ["cup", "dog"], 100, 100, threshold=0.3)
    # The lower cup overlaps the better one; one cup is below the threshold.
    assert boxes == [[[0, 0, 10, 9]], [[0, 0, 10, 10]]] and scores == [[0.9], [0.8]]
    boxes, _ = select(found, ["cup"], 100, 100, threshold=0.3, duplicate_iou=None)
    assert boxes == [[[0, 0, 10, 9], [0, 0, 10, 10]]]


def test_fuse_averages_overlapping_boxes_by_score():
    found = [candidate([0, 0, 10, 10], "cup", 0.75), candidate([2, 0, 12, 10], "cup", 0.25)]
    boxes, scores = select(found, ["cup"], 100, 100, 0.1, duplicate_iou=0.5, merge="fuse")
    assert boxes == [[[0.5, 0.0, 10.5, 10.0]]] and scores == [[0.75]]


def test_aliases_and_tile_candidates():
    found = [
        candidate([0, 0, 50, 50], "car", 0.9),
        candidate([60, 60, 90, 90], "truck", 0.8),
        candidate([1, 1, 3, 3], "car", 0.7, view="tile"),
        candidate([5, 5, 7, 7], "car", 0.7, view="tile", cut=True),
        candidate([0, 0, 60, 60], "car", 0.6, view="tile"),
    ]
    width = height = 270  # ten pixels per grid cell
    assert cells([0, 0, 20, 20], width, height) == 2.0
    boxes, _ = select(found, ["car"], width, height, 0.5, aliases={"car": ["car", "truck"]})
    assert boxes == [[[0, 0, 50, 50], [60, 60, 90, 90]]]  # tiles off by default
    boxes, _ = select(found, ["car"], width, height, 0.5, tile_max_cells=4)
    # The small uncut tile box counts; the cut one and the large one do not.
    assert boxes == [[[0, 0, 50, 50], [1, 1, 3, 3]]]


def test_previous_boxes_lower_the_threshold_for_continuing_objects():
    found = [candidate([0, 0, 10, 10], "car", 0.4), candidate([50, 50, 60, 60], "car", 0.4)]
    boxes, _ = select(found, ["car"], 100, 100, threshold=0.5)
    assert boxes == [[]]
    previous = {"car": [[1, 0, 11, 10]]}
    boxes, _ = select(found, ["car"], 100, 100, 0.5, previous=previous, keep_threshold=0.3)
    # Only the candidate continuing last frame's car passes the lower threshold.
    assert boxes == [[[0, 0, 10, 10]]]
