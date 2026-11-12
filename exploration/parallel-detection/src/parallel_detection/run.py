"""Run a detector over the configured tasks and write predictions and a summary.

    python -m parallel_detection.run evaluate --config configs/baseline.json --out artifacts/RUN
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
from parallel_detection.tasks import lvis_samples, refcoco_samples


def build_samples(config, download):
    """`download(dataset)` returns the local path of that dataset's annotation file."""
    samples = []
    for dataset in config["datasets"]:
        path = download(dataset)
        if dataset["kind"] == "lvis":
            built = lvis_samples(
                json.loads(Path(path).read_text()),
                dataset["images"],
                dataset["max_absent"],
                config["seed"],
            )
        elif dataset["kind"] == "refcoco":
            import pyarrow.parquet as pq

            columns = ["ref_id", "image_id", "sentences", "raw_anns", "raw_image_info"]
            rows = pq.read_table(path, columns=columns).to_pylist()
            built = refcoco_samples(rows, dataset["refs"], config["seed"])
        else:
            raise ValueError(f"Unknown dataset kind {dataset['kind']!r}")
        samples += [sample | {"dataset": dataset["name"]} for sample in built]
    return samples


def hub_download(dataset):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        dataset["repo"],
        dataset["file"],
        repo_type=dataset["repo_type"],
        revision=dataset["revision"],
    )


def fetch_image(url, folder):
    """Download once into `folder`; COCO serves these images over plain HTTP only."""
    from PIL import Image

    path = Path(folder) / url.split("://", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        Image.open(io.BytesIO(data)).verify()
        path.write_bytes(data)
    return Image.open(path).convert("RGB")


def predict(detector, samples, image_folder, warmup=2):
    """Yield each sample's records, one per query.

    The first `warmup` images run twice; only the second run is recorded.
    """
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda s: fetch_image(s["image_url"], image_folder), samples))
    for index, sample in enumerate(samples):
        image = fetch_image(sample["image_url"], image_folder)
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


def evaluate(config, out, device, provenance=None, saved=lambda images: None):
    """Write `predictions.jsonl` and `summary.json` under `out`; return the summary.

    Predictions are flushed image by image and `saved(images)` is called after each,
    so an interrupted run keeps what it finished.
    """
    from parallel_detection.moondream import MoondreamDetector

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples = build_samples(config, hub_download)
    detector = MoondreamDetector(**config["model"], device=device)
    records = []
    with (out / "predictions.jsonl").open("w") as handle:
        for done, batch in enumerate(predict(detector, samples, out.parent / "images"), 1):
            records += batch
            handle.writelines(json.dumps(record) + "\n" for record in batch)
            handle.flush()
            saved(done)
    summary = {
        "config": config,
        "environment": environment(device),
        "provenance": provenance or {},
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
        records = [json.loads(line) for line in args.predictions.read_text().splitlines()]
        print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
