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
    """(label, box, score) triples from a detector result for `phrases`; the score is
    None for detectors that give no confidence."""
    scores = result.get("scores") or [[None] * len(boxes) for boxes in result["boxes"]]
    return [
        (phrase, box, score)
        for phrase, boxes, confidences in zip(phrases, result["boxes"], scores, strict=True)
        for box, score in zip(boxes, confidences, strict=True)
    ]


def draw(image, found, caption=None):
    """A copy of `image` with each detection boxed and labelled with its score, and an
    optional caption."""
    image = image.convert("RGB").copy()
    pen = ImageDraw.Draw(image)
    width = max(2, round(max(image.size) / 400))
    for label, (x0, y0, x1, y1), score in found:
        hue = colour(label)
        text = label if score is None else f"{label} {score:.2f}"
        pen.rectangle([x0, y0, x1, y1], outline=hue, width=width)
        left, top, right, bottom = pen.textbbox((x0, y0), text)
        pen.rectangle([left - 2, top - 2, right + 2, bottom + 2], fill=hue)
        pen.text((x0, y0), text, fill="white")
    if caption:
        left, top, right, bottom = pen.textbbox((6, 6), caption)
        pen.rectangle([0, 0, right + 8, bottom + 6], fill="black")
        pen.text((6, 6), caption, fill="white")
    return image


def _encode(frames, size, fps):
    """H.264 in an MPEG transport stream: the chunk format streaming video players take."""
    import tempfile

    import imageio_ffmpeg

    with tempfile.NamedTemporaryFile(suffix=".ts") as chunk:
        writer = imageio_ffmpeg.write_frames(
            chunk.name, size, fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=7
        )
        writer.send(None)
        for frame in frames:
            writer.send(frame)
        writer.close()
        return chunk.read()


def stream_video(data, detect, every, max_frames, chunk_seconds=2.0, track=False):
    """Run `detect(image) -> [(label, box, score)]` on every `every`-th frame of a video.

    Frames in between show the latest detections. With `track`, detections pass through
    `tracking.Tracker` (smoothed boxes, one missed detection coasted) and are labelled
    with their track id. Yields (chunk, statistics) about every
    `chunk_seconds` of video: the annotated chunk as H.264 MPEG-TS bytes, and statistics
    on frames, detector calls and seconds so far.
    """
    import tempfile

    import imageio_ffmpeg

    # Imported here: the web page uses this module without the tracker's dependencies.
    from parallel_detection.tracking import Tracker

    with tempfile.NamedTemporaryFile(suffix=".mp4") as clip:
        clip.write(data)
        clip.flush()
        reader = imageio_ffmpeg.read_frames(clip.name)
        meta = next(reader)
        size, fps = meta["size"], meta.get("fps") or 25.0
        per_chunk = max(1, round(fps * chunk_seconds))
        found, seconds, pending, frames = [], [], [], 0
        tracker = Tracker(smoothing=0.5, coast=1) if track else None
        began = time.perf_counter()
        try:
            for raw in reader:
                if frames == max_frames:
                    break
                frames += 1
                image = Image.frombytes("RGB", size, raw)
                if (frames - 1) % every == 0:
                    start = time.perf_counter()
                    found = detect(image)
                    if tracker is not None:
                        tracks = tracker.update(
                            [b for _, b, _ in found], [s for *_, s in found], [n for n, *_ in found]
                        )
                        found = [(f"{t['label']} #{t['id']}", t["box"], t["score"]) for t in tracks]
                    seconds.append(time.perf_counter() - start)
                pending.append(draw(image, found).tobytes())
                if len(pending) == per_chunk:
                    yield _encode(pending, size, fps), _stats(frames, every, fps, seconds, began)
                    pending = []
            if pending:
                yield _encode(pending, size, fps), _stats(frames, every, fps, seconds, began)
        finally:
            reader.close()


def _stats(frames, every, fps, seconds, began):
    total = sum(seconds)
    wall = time.perf_counter() - began
    return {
        "frames": frames,
        "detector_calls": len(seconds),
        "every": every,
        "video_fps": fps,
        "seconds_per_call": total / len(seconds) if seconds else None,
        "detector_seconds": total,
        # Above 1 the stream keeps up with the video's own frame rate.
        "speed_vs_real_time": (frames / fps) / wall if wall else None,
    }
