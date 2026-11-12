import pytest

from parallel_detection.boxes import iou_matrix, match
from parallel_detection.metrics import agreement, summarize, unit_confidence_ap


def record(kind, truth, predicted, exhaustive=True, image_id=1, seconds=0.1):
    return {
        "dataset": "d",
        "image_id": image_id,
        "phrase": "thing",
        "kind": kind,
        "boxes": truth,
        "exhaustive": exhaustive,
        "group": "f",
        "predicted": predicted,
        "seconds": seconds,
        "shared_seconds": 0.5,
    }


def test_iou():
    iou = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert iou[0].tolist() == pytest.approx([1.0, 1 / 3, 0.0])


def test_match_takes_the_most_pairs_not_the_best_single_pair():
    # Prediction 0 overlaps both truths; prediction 1 only truth 0. A greedy best-IoU
    # match of prediction 0 to truth 0 would leave prediction 1 unmatched.
    predicted = [[0, 0, 10, 10], [0, 0, 10, 7]]
    truth = [[0, 0, 10, 9], [0, 1, 10, 12]]
    assert sorted(match(predicted, truth, 0.6)) == [(0, 1), (1, 0)]
    assert match([], truth, 0.5) == [] and match(predicted, [], 0.5) == []


def test_detection_counts():
    box, far = [0, 0, 100, 100], [500, 500, 520, 520]
    records = [
        record("present", [box, far], [box, [300, 300, 310, 310]]),  # 1 tp, 1 fp, 1 fn
        record("present", [box], [box, far], exhaustive=False),  # 1 tp, extra box not an error
        record("absent", [], [far]),  # 1 fp
        record("absent", [], []),
    ]
    detection = summarize(records)["d"]["detection"]
    half = detection["iou_0.5"]
    assert (half["tp"], half["fp"], half["fn"]) == (2, 2, 1)
    assert half["recall"] == pytest.approx(2 / 3)
    assert half["recall_by_size"]["large"] == {"recall": 1.0, "truth_boxes": 2}
    assert half["recall_by_size"]["small"] == {"recall": 0.0, "truth_boxes": 1}
    assert half["recall_by_instances_in_query"]["2-5"]["recall"] == 0.5
    assert half["phrase_hit_rate"] == 1.0
    assert detection["absent"] == {"queries": 2, "false_alarm_rate": 0.5, "boxes_per_query": 0.5}


def test_unit_confidence_ap_follows_emitted_order():
    box, far, miss = [0, 0, 100, 100], [500, 500, 520, 520], [300, 300, 310, 310]
    assert unit_confidence_ap([box, far], [box, far], 0.5) == 1.0
    assert unit_confidence_ap([], [box], 0.5) == 0.0
    # One of two truths found by the first box: precision 1 up to recall 0.5, then 0.
    assert unit_confidence_ap([box, miss], [box, far], 0.5) == pytest.approx(6 / 11)
    # The same boxes with the miss first: precision 0.5 up to recall 0.5.
    assert unit_confidence_ap([miss, box], [box, far], 0.5) == pytest.approx(3 / 11)


def test_referring_scores_the_first_box():
    box, other = [0, 0, 100, 100], [200, 200, 300, 300]
    records = [
        record("referring", [box], [box]),
        record("referring", [box], [other, box]),
        record("referring", [box], []),
    ]
    summary = summarize(records)["d"]
    assert summary["detection"]["iou_0.5"]["tp"] == 2  # F1 counts any box, not only the first
    referring = summary["referring"]
    assert referring["accuracy_iou_0.5"] == pytest.approx(1 / 3)
    assert referring["one_box_rate"] == pytest.approx(1 / 3)
    assert referring["no_box_rate"] == pytest.approx(1 / 3)


def test_latency_counts_shared_work_once_per_image():
    box = [0, 0, 10, 10]
    records = [
        record("present", [box], [], image_id=1, seconds=0.1),
        record("present", [box], [box, box], image_id=1, seconds=0.3),
        record("present", [box], [box], image_id=2, seconds=0.2),
    ]
    latency = summarize(records)["d"]["latency"]
    assert latency["image_seconds"]["mean"] == pytest.approx((0.9 + 0.7) / 2)
    assert latency["phrase_seconds_linear_fit"]["seconds_per_box"] == pytest.approx(0.1)
    assert latency["phrase_seconds_by_boxes_emitted"]["0"]["n"] == 1


def test_agreement_pairs_boxes_in_emitted_order():
    box, shifted = [0, 0, 10, 10], [0, 0, 10, 5]
    reference = [record("present", [box], [box]), record("present", [box], [box, box])]
    other = [record("present", [box], [shifted]), record("present", [box], [box])]
    result = agreement(reference, other)
    assert result["same_box_count_rate"] == 0.5
    assert result["identical_boxes_rate"] == 0.0
    assert result["paired_iou"]["mean"] == pytest.approx(0.5)
