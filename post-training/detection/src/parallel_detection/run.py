"""Run a detector over the configured tasks and write predictions and a summary.

    python -m parallel_detection.run evaluate --config configs/moondream2.json --out artifacts/RUN
    python -m parallel_detection.run score --predictions artifacts/RUN/predictions.jsonl

A detector is any object with `detect(image, phrases)` returning per-phrase pixel
boxes, the seconds spent once per image and the seconds spent per phrase. A model
that answers every phrase in one pass reports that pass as `shared_seconds`.
"""

import argparse
import io
import json
import platform
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from parallel_detection.metrics import summarize
from parallel_detection.tasks import coco_samples, lvis_samples, refcoco_m_samples


def hub_files(dataset):
    from huggingface_hub import hf_hub_download

    return [
        hf_hub_download(
            dataset["repo"], name, repo_type=dataset["repo_type"], revision=dataset["revision"]
        )
        for name in dataset["files"]
    ]


def parquet_rows(paths, columns):
    import pyarrow.parquet as pq

    return [row for path in paths for row in pq.read_table(path, columns=columns).to_pylist()]


def extract_images(paths, samples, image_folder):
    """Write the sampled images embedded in Parquet files to the image folder."""
    import pyarrow.parquet as pq

    wanted = {s["image_id"]: Path(image_folder) / s["image"] for s in samples}
    for path in paths:
        table = pq.ParquetFile(path)
        for group in range(table.metadata.num_row_groups):
            ids = table.read_row_group(group, columns=["image_id"]).column("image_id").to_pylist()
            if not any(i in wanted and not wanted[i].exists() for i in ids):
                continue
            images = table.read_row_group(group, columns=["image"]).column("image").to_pylist()
            for image_id, image in zip(ids, images, strict=True):
                if image_id in wanted:
                    wanted[image_id].parent.mkdir(parents=True, exist_ok=True)
                    wanted[image_id].write_bytes(image["bytes"])


def build_samples(config, image_folder, files=hub_files):
    """`files(dataset)` returns the local paths of that dataset's files."""
    samples = []
    for dataset in config["datasets"]:
        paths = files(dataset)
        if dataset["kind"] == "lvis":
            built = lvis_samples(
                json.loads(Path(paths[0]).read_text()),
                dataset["images"],
                dataset["max_absent"],
                config["seed"],
            )
        elif dataset["kind"] == "coco":
            rows = parquet_rows(paths, ["image_id", "width", "height", "objects"])
            built = coco_samples(rows, dataset["images"], config["seed"])
            extract_images(paths, built, image_folder)
        elif dataset["kind"] == "refcoco_m":
            rows = parquet_rows(paths, ["image_id", "samples"])
            built = refcoco_m_samples(rows, dataset["images"], config["seed"])
            extract_images(paths, built, image_folder)
        else:
            raise ValueError(f"Unknown dataset kind {dataset['kind']!r}")
        samples += [sample | {"dataset": dataset["name"]} for sample in built]
    return samples


def fetch_image(reference, folder):
    """Open an image from the folder, downloading a URL into it first.

    COCO serves its images over plain HTTP only.
    """
    from PIL import Image

    if "://" not in reference:
        return Image.open(Path(folder) / reference).convert("RGB")
    path = Path(folder) / reference.split("://", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(reference, timeout=60) as response:
            data = response.read()
        Image.open(io.BytesIO(data)).verify()
        path.write_bytes(data)
    return Image.open(path).convert("RGB")


def predict(detector, samples, image_folder, warmup=2):
    """Yield each sample's records, one per query.

    The first `warmup` images run twice; only the second run is recorded.
    """
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda s: fetch_image(s["image"], image_folder), samples))
    for index, sample in enumerate(samples):
        image = fetch_image(sample["image"], image_folder)
        phrases = [q["phrase"] for q in sample["queries"]]
        if index < warmup:
            detector.detect(image, phrases)
        result = detector.detect(image, phrases)
        yield [
            query
            | {
                "dataset": sample["dataset"],
                "image_id": sample["image_id"],
                "predicted": boxes,
                "seconds": seconds,
                "shared_seconds": result["shared_seconds"],
            }
            for query, boxes, seconds in zip(
                sample["queries"], result["boxes"], result["phrase_seconds"], strict=True
            )
        ]


def environment(device):
    import torch
    import transformers

    return {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def evaluate(config, out, device, provenance=None, saved=lambda images: None, **detector):
    """Write `predictions.jsonl` and `summary.json` under `out`; return the summary.

    Predictions are appended image by image and `saved(images)` is called after each.
    A run interrupted before its summary continues from the images already recorded.
    `detector` holds extra `MoondreamDetector` arguments.
    """
    from parallel_detection.moondream import MoondreamDetector

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    predictions = out / "predictions.jsonl"
    records = read_records(predictions) if predictions.exists() else []
    finished = {(r["dataset"], r["image_id"]) for r in records}
    samples = build_samples(config, out.parent / "images")
    remaining = [s for s in samples if (s["dataset"], s["image_id"]) not in finished]
    detector = MoondreamDetector(**config["model"], device=device, **detector)
    with predictions.open("a") as handle:
        for done, batch in enumerate(predict(detector, remaining, out.parent / "images"), 1):
            records += batch
            handle.writelines(json.dumps(record) + "\n" for record in batch)
            handle.flush()
            saved(done)
    summary = {
        "config": config,
        "environment": environment(device),
        "provenance": provenance or {},
        "resumed_after_images": len(samples) - len(remaining),
        "datasets": summarize(records),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    score = commands.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        summary = evaluate(json.loads(args.config.read_text()), args.out, args.device)
        print(json.dumps(summary["datasets"], indent=2))
    else:
        print(json.dumps(summarize(read_records(args.predictions)), indent=2))


if __name__ == "__main__":
    main()
