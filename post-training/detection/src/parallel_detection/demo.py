"""Drawing and video helpers for the detection demo (`demo_app.py`).

A detection is (label, [x0, y0, x1, y1]) in pixels of the image it was found in.
"""

import hashlib
import time

from PIL import Image, ImageDraw

from parallel_detection.tasks import COCO_NAMES

# Colour per category: a fixed hue order cycled by a stable hash of the name.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def colour(label):
    return PALETTE[int(hashlib.md5(label.encode()).hexdigest(), 16) % len(PALETTE)]


def phrases_of(text):
    """Categories asked for in a comma-separated text; empty means all 80 COCO ones."""
    asked = [p.strip().lower() for p in text.split(",") if p.strip()]
    return asked or list(COCO_NAMES)


def detections(result, phrases):
    """(label, box) pairs from a detector result for `phrases`."""
    return [
        (phrase, box)
        for phrase, boxes in zip(phrases, result["boxes"], strict=True)
        for box in boxes
    ]


def draw(image, found, caption=None):
    """A copy of `image` with each detection boxed and labelled, and an optional caption."""
    image = image.convert("RGB").copy()
    pen = ImageDraw.Draw(image)
    width = max(2, round(max(image.size) / 400))
    for label, (x0, y0, x1, y1) in found:
        hue = colour(label)
        pen.rectangle([x0, y0, x1, y1], outline=hue, width=width)
        left, top, right, bottom = pen.textbbox((x0, y0), label)
        pen.rectangle([left - 2, top - 2, right + 2, bottom + 2], fill=hue)
        pen.text((x0, y0), label, fill="white")
    if caption:
        left, top, right, bottom = pen.textbbox((6, 6), caption)
        pen.rectangle([0, 0, right + 8, bottom + 6], fill="black")
        pen.text((6, 6), caption, fill="white")
    return image


def run_video(data, detect, every, max_frames):
    """Run `detect(image) -> [(label, box)]` on every `every`-th frame of a video.

    Frames in between show the latest detections. Returns the annotated H.264 MP4 and
    statistics on frames, detector calls and detector seconds.
    """
    import tempfile

    import imageio_ffmpeg

    with (
        tempfile.NamedTemporaryFile(suffix=".mp4") as clip,
        tempfile.NamedTemporaryFile(suffix=".mp4") as result,
    ):
        clip.write(data)
        clip.flush()
        reader = imageio_ffmpeg.read_frames(clip.name)
        meta = next(reader)
        size, fps = meta["size"], meta.get("fps") or 25.0
        writer = imageio_ffmpeg.write_frames(
            result.name, size, fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=7
        )
        writer.send(None)
        found, calls, seconds, frames = [], 0, [], 0
        for frames, raw in enumerate(reader, 1):
            if frames > max_frames:
                break
            image = Image.frombytes("RGB", size, raw)
            if (frames - 1) % every == 0:
                start = time.perf_counter()
                found = detect(image)
                seconds.append(time.perf_counter() - start)
                calls += 1
            writer.send(draw(image, found).tobytes())
        writer.close()
        reader.close()
        with open(result.name, "rb") as handle:
            output = handle.read()
    total = sum(seconds)
    stats = {
        "frames": min(frames, max_frames),
        "detector_calls": calls,
        "every": every,
        "video_fps": fps,
        "seconds_per_call": total / calls if calls else None,
        "detector_seconds": total,
        "frames_per_second_covered": (min(frames, max_frames) / total) if total else None,
    }
    return output, stats
