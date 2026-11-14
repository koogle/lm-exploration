from parallel_detection.teacher import consensus, parse


def test_parse_reads_gemini_style_boxes_and_drops_unknown_labels():
    text = (
        "```json\n"
        '{"objects": [{"label": "Person", "box_2d": [100, 200, 500, 400]},'
        ' {"label": "unicorn", "box_2d": [0, 0, 10, 10]},'
        ' {"label": "cup", "box_2d": [500, 500, 400, 600]}]}\n```'
    )
    assert parse(text, width=1000, height=500) == [
        {"label": "person", "box": [200.0, 50.0, 400.0, 250.0]}
    ]
    assert parse("no json here", 10, 10) == [] and parse(None, 10, 10) == []


def test_consensus_keeps_boxes_enough_models_agree_on():
    person = {"label": "person", "box": [0, 0, 100, 100]}
    shifted = {"label": "person", "box": [5, 0, 105, 100]}
    cup = {"label": "cup", "box": [200, 200, 220, 220]}
    by_model = {
        "a": [person, cup],
        "b": [shifted],
        "c": [{"label": "cup", "box": [0, 0, 100, 100]}],
    }
    kept = consensus(by_model, votes=2)
    assert kept == [{"label": "person", "box": [2.5, 0.0, 102.5, 100.0]}]
    assert len(consensus(by_model, votes=1)) == 3


def test_save_resizes_and_skips_existing_images(tmp_path):
    import io

    from PIL import Image

    from parallel_detection.images import save

    buffer = io.BytesIO()
    Image.new("RGB", (2048, 1024)).save(buffer, format="JPEG")
    line = save(buffer.getvalue(), tmp_path, "openimages", "7-3", 1024, existing=set())
    assert line == {
        "image_id": "openimages/7-3",
        "image": "openimages/7-3.jpg",
        "width": 1024,
        "height": 512,
        "source": "openimages",
    }
    assert save(buffer.getvalue(), tmp_path, "openimages", "7-3", 1024, {"openimages/7-3"}) is None
    assert save(b"not an image", tmp_path, "openimages", "x", 1024, set()) is None
