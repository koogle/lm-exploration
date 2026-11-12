"""Unlabelled training images from Open Images and Objects365, for teacher labelling.

    python -m parallel_detection.images --config configs/teacher-images.json \
        --out artifacts/teacher-images

Writes `<out>/<source>/<id>.jpg`, with the longest side at most `max_side`, and appends
one line per image to `<out>/manifest.jsonl` ({"image_id", "image", "width", "height",
"source"}, with "image" relative to `<out>`). A source that already has its quota in
the manifest is skipped, so an interrupted run continues.

- Open Images (`vikhyatk/openimages-bbox`, Parquet): `per_shard` seeded images from
  each of `shards` seeded training shards, one shard on disk at a time.
- Objects365 (`guozonghao96/objects365`, tar.gz patches): every `stride`-th image of
  each listed patch, streamed without saving the archive, until the quota is reached.
"""

import argparse
import io
import json
import random
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

HUB = "https://huggingface.co/datasets"


def save(data, out, source, name, max_side, existing):
    """Write a resized RGB JPEG; return its manifest line, or None if unreadable or
    already in the manifest."""
    if f"{source}/{name}" in existing:
        return None
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (OSError, ValueError):
        return None
    image.thumbnail((max_side, max_side))
    path = Path(source) / f"{name}.jpg"
    (out / path).parent.mkdir(parents=True, exist_ok=True)
    image.save(out / path, format="JPEG", quality=90)
    width, height = image.size
    existing.add(f"{source}/{name}")
    return {
        "image_id": f"{source}/{name}",
        "image": str(path),
        "width": width,
        "height": height,
        "source": source,
    }


def open_images(settings, out, quota, max_side, rng, existing):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    shards = rng.sample(range(settings["total_shards"]), settings["shards"])
    for shard in sorted(shards):
        if quota <= 0:
            return
        name = f"data/train-{shard:05d}-of-{settings['total_shards']:05d}.parquet"
        path = Path(
            hf_hub_download(
                settings["repo"],
                name,
                repo_type="dataset",
                revision=settings["revision"],
                cache_dir=out / ".download",
            )
        )
        table = pq.read_table(path, columns=["image"])
        rows = rng.sample(range(table.num_rows), min(settings["per_shard"], table.num_rows, quota))
        images = table.column("image")
        for row in sorted(rows):
            line = save(
                images[row].as_py()["bytes"],
                out,
                "openimages",
                f"{shard}-{row}",
                max_side,
                existing,
            )
            if line:
                quota -= 1
                yield line
        # The cache entry is a symlink to a blob; remove both to keep one shard on disk.
        path.resolve().unlink()
        path.unlink()


def objects365(settings, out, quota, max_side, existing):
    for patch in settings["patches"]:
        url = f"{HUB}/{settings['repo']}/resolve/{settings['revision']}/{patch}"
        # A dropped connection restarts the patch; images already saved are skipped.
        for _attempt in range(4):
            if quota <= 0:
                return
            try:
                for line in patch_images(url, settings["stride"], out, max_side, existing):
                    quota -= 1
                    yield line
                    if quota <= 0:
                        return
                break
            except (tarfile.ReadError, OSError, EOFError):
                continue


def patch_images(url, stride, out, max_side, existing):
    """Every `stride`-th image of one streamed tar.gz patch, saved and resized."""
    with urllib.request.urlopen(url, timeout=600) as response:
        seen = 0
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".jpg"):
                    continue
                seen += 1
                if seen % stride:
                    continue
                data = archive.extractfile(member).read()
                line = save(data, out, "objects365", Path(member.name).stem, max_side, existing)
                if line:
                    yield line


def prepare(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    counts, existing = {}, set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            record = json.loads(line)
            counts[record["source"]] = counts.get(record["source"], 0) + 1
            existing.add(record["image_id"])
    rng = random.Random(config["seed"])
    makers = {
        "openimages": lambda s, q: open_images(s, out, q, config["max_side"], rng, existing),
        "objects365": lambda s, q: objects365(s, out, q, config["max_side"], existing),
    }
    for source, settings in config["sources"].items():
        quota = settings["images"] - counts.get(source, 0)
        if quota <= 0:
            continue
        with manifest.open("a") as handle:
            for line in makers[source](settings, quota):
                handle.write(json.dumps(line) + "\n")
                handle.flush()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(json.loads(args.config.read_text()), args.out))


if __name__ == "__main__":
    main()
