import json

from parallel_detection.run import predict
from parallel_detection.tasks import lvis_samples, refcoco_samples

LVIS = {
    "categories": [
        {"id": 1, "name": "bow_(weapon)", "frequency": "r"},
        {"id": 2, "name": "cup", "frequency": "f"},
        {"id": 3, "name": "zebra", "frequency": "c"},
        {"id": 4, "name": "kite", "frequency": "c"},
    ],
    "images": [
        {
            "id": 7,
            "coco_url": "http://images.cocodataset.org/val2017/7.jpg",
            "width": 640,
            "height": 480,
            "neg_category_ids": [3, 4],
            "not_exhaustive_category_ids": [2],
        },
        {
            "id": 8,
            "coco_url": "http://images.cocodataset.org/val2017/8.jpg",
            "width": 640,
            "height": 480,
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        },
    ],
    "annotations": [
        {"image_id": 7, "category_id": 1, "bbox": [10, 20, 30, 40]},
        {"image_id": 7, "category_id": 2, "bbox": [0, 0, 5, 5]},
        {"image_id": 7, "category_id": 2, "bbox": [50, 50, 5, 5]},
    ],
}


def test_lvis_prompts_present_and_verified_absent_categories():
    (sample,) = lvis_samples(LVIS, images=5, max_absent=1, seed=0)  # image 8 has no boxes
    assert sample["image_id"] == 7
    bow, cup, absent = sample["queries"]
    assert (bow["phrase"], bow["boxes"], bow["exhaustive"]) == (
        "bow (weapon)",
        [[10, 20, 40, 60]],
        True,
    )
    assert (len(cup["boxes"]), cup["exhaustive"], cup["group"]) == (2, False, "f")
    assert absent["kind"] == "absent" and absent["phrase"] in {"zebra", "kite"}
    assert lvis_samples(LVIS, 5, 1, seed=0) == lvis_samples(LVIS, 5, 1, seed=0)


def test_refcoco_groups_referred_objects_by_image():
    rows = [
        {
            "ref_id": ref,
            "image_id": 99,
            "sentences": [{"raw": f"object {ref} "}],
            "raw_anns": json.dumps({"bbox": [ref, 2, 3, 4]}),
            "raw_image_info": json.dumps(
                {"file_name": "COCO_train2014_000000000099.jpg", "width": 640, "height": 428}
            ),
        }
        for ref in (1, 2)
    ]
    (sample,) = refcoco_samples(rows, refs=10, seed=0)
    assert sample["image_url"].endswith("/train2014/COCO_train2014_000000000099.jpg")
    assert sorted(q["phrase"] for q in sample["queries"]) == ["object 1", "object 2"]
    assert all(q["kind"] == "referring" and len(q["boxes"]) == 1 for q in sample["queries"])


def test_predict_joins_detections_to_queries(tmp_path):
    from PIL import Image

    (sample,) = lvis_samples(LVIS, images=5, max_absent=0, seed=0)
    sample["dataset"] = "lvis_minival"
    cached = tmp_path / "images.cocodataset.org/val2017/7.jpg"
    cached.parent.mkdir(parents=True)
    Image.new("RGB", (64, 48)).save(cached)

    class Detector:
        calls = 0

        def detect(self, image, phrases):
            self.calls += 1
            return {
                "boxes": [[[0, 0, 1, 1]] for _ in phrases],
                "shared_seconds": 0.5,
                "phrase_seconds": [0.1] * len(phrases),
            }

    detector = Detector()
    (records,) = predict(detector, [sample], tmp_path, warmup=1)
    assert detector.calls == 2  # the warm-up pass is not recorded
    assert [r["phrase"] for r in records] == ["bow (weapon)", "cup"]
    assert all(r["predicted"] == [[0, 0, 1, 1]] and r["dataset"] == "lvis_minival" for r in records)
