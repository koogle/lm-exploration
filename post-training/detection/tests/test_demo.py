import pytest

from parallel_detection.demo import detections, draw, phrases_of, run_video
from parallel_detection.tasks import COCO_NAMES


def test_phrases_and_detections():
    assert phrases_of(" Person , cup,") == ["person", "cup"]
    assert phrases_of("") == list(COCO_NAMES)
    result = {"boxes": [[[0, 0, 5, 5]], [], [[1, 1, 2, 2], [3, 3, 4, 4]]]}
    assert detections(result, ["a", "b", "c"]) == [
        ("a", [0, 0, 5, 5]),
        ("c", [1, 1, 2, 2]),
        ("c", [3, 3, 4, 4]),
    ]


def test_draw_keeps_size_and_marks_boxes():
    from PIL import Image

    image = Image.new("RGB", (64, 48))
    drawn = draw(image, [("person", [4, 4, 40, 40])], caption="model, 0.1 s")
    assert drawn.size == image.size and drawn.getbbox() is not None
    assert image.getbbox() is None  # the input is not modified


def test_run_video_detects_every_nth_frame(tmp_path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "clip.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(path), (64, 48), fps=10, codec="libx264", pix_fmt_out="yuv420p"
    )
    writer.send(None)
    for index in range(10):
        writer.send(bytes([index * 20 % 256]) * (64 * 48 * 3))
    writer.close()
    seen = []

    def detect(image):
        seen.append(image.size)
        return [("cup", [1, 1, 20, 20])]

    output, stats = run_video(path.read_bytes(), detect, every=3, max_frames=100)
    assert seen == [(64, 48)] * 4  # frames 1, 4, 7 and 10
    assert stats["frames"] == 10 and stats["detector_calls"] == 4 and output[4:8] == b"ftyp"
