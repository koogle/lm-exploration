from parallel_detection.tracking import Tracker


def test_tracks_keep_ids_smooth_boxes_and_coast_one_frame():
    tracker = Tracker(match_iou=0.3, smoothing=0.5, coast=1)
    first = tracker.update([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.8], ["person", "car"])
    ids = {t["label"]: t["id"] for t in first}
    second = tracker.update([[2, 0, 12, 10]], [0.7], ["person"])
    person = next(t for t in second if t["label"] == "person")
    assert person["id"] == ids["person"] and person["box"] == [1.0, 0.0, 11.0, 10.0]
    # The car went unseen for one frame: it is still reported, marked as coasting.
    car = next(t for t in second if t["label"] == "car")
    assert car["coasting"] and car["id"] == ids["car"]
    third = tracker.update([], [], [])
    assert [t["label"] for t in third] == ["person"]  # the car has coasted out
