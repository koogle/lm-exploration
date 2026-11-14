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
from parallel_detection.tasks import (
    coco_samples,
    lvis_samples,
    mot_samples,
    refcoco_m_samples,
    voc_annotation,
    voc_samples,
)


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
            built = coco_samples(
                rows,
                dataset["images"],
                config["seed"],
                exclude_images=dataset.get("exclude_images", 0),
            )
            extract_images(paths, built, image_folder)
        elif dataset["kind"] == "refcoco_m":
            rows = parquet_rows(paths, ["image_id", "samples"])
            built = refcoco_m_samples(rows, dataset["images"], config["seed"])
            extract_images(paths, built, image_folder)
        elif dataset["kind"] == "mot":
            built = mot_dataset_samples(dataset, image_folder)
        elif dataset["kind"] == "voc":
            built = voc_tar_samples(paths[0], dataset, config["seed"], image_folder)
        else:
            raise ValueError(f"Unknown dataset kind {dataset['kind']!r}")
        samples += [sample | {"dataset": dataset["name"]} for sample in built]
    return samples


def mot_dataset_samples(dataset, image_folder):
    """Samples of MOT17 sequences from a Hub copy, with the sampled frames written to the
    image folder."""
    import configparser
    import shutil

    from huggingface_hub import hf_hub_download

    def fetch(name):
        return hf_hub_download(
            dataset["repo"], name, repo_type="dataset", revision=dataset["revision"]
        )

    built = []
    for sequence in dataset["sequences"]:
        folder = f"{dataset['split']}/{sequence}"
        info = configparser.ConfigParser()
        info.read(fetch(f"{folder}/seqinfo.ini"))
        width, height = int(info["Sequence"]["imWidth"]), int(info["Sequence"]["imHeight"])
        truth = Path(fetch(f"{folder}/gt/gt.txt")).read_text()
        samples = mot_samples(sequence, truth, width, height, dataset["every"])
        for sample in samples:
            target = Path(image_folder) / sample["image"]
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                frame = sample["image_id"].split("/")[1]
                shutil.copyfile(fetch(f"{folder}/img1/{frame}.jpg"), target)
        built += samples
    return built


def voc_tar_samples(path, dataset, seed, image_folder):
    """Samples from a Pascal VOC tar: annotations of the images listed in
    `dataset["image_set"]` (a path inside the tar), and their JPEGs written to the
    image folder."""
    import tarfile

    with tarfile.open(path) as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        listed = tar.extractfile(members[dataset["image_set"]]).read().decode().split()
        root = dataset["image_set"].split("/ImageSets/")[0]
        annotations = [
            voc_annotation(tar.extractfile(members[f"{root}/Annotations/{name}.xml"]).read())
            for name in listed
        ]
        built = voc_samples(annotations, dataset["images"], seed)
        for sample in built:
            target = Path(image_folder) / sample["image"]
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                jpeg = members[f"{root}/JPEGImages/{sample['image_id']}.jpg"]
                target.write_bytes(tar.extractfile(jpeg).read())
    return built


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
                "components": result.get("components"),
            }
            | ({"scores": result["scores"][position]} if "scores" in result else {})
            # A detector that reports every slot attaches them to the image's first query.
            | ({"slots": result["slots"]} if "slots" in result and position == 0 else {})
            for position, (query, boxes, seconds) in enumerate(
                zip(sample["queries"], result["boxes"], result["phrase_seconds"], strict=True)
            )
        ]


def environment(device):
    from importlib import metadata

    import torch

    versions = {}
    for package in ("torch", "transformers", "moondream", "kestrel", "kestrel-kernels"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "python": platform.python_version(),
        "packages": versions,
    }


def make_detector(model, device, **extra):
    """Build the detector a config's "model" names; its "adapter" picks the runtime."""
    model = dict(model)
    adapter = model.pop("adapter")
    if adapter == "transformers":
        from parallel_detection.moondream import MoondreamDetector

        return MoondreamDetector(**model, device=device, **extra)
    if adapter == "photon":
        from parallel_detection.photon import PhotonDetector

        return PhotonDetector(**model, device=device)
    if adapter == "slots":
        from parallel_detection.train_slots import SlotDetector

        return SlotDetector(**model, device=device)
    raise ValueError(f"Unknown adapter {adapter!r}")


def above(records, threshold):
    """Records keeping only the boxes whose recorded confidence reaches `threshold`."""
    return [
        r
        | {
            "predicted": [
                box
                for box, score in zip(r["predicted"], r["scores"], strict=True)
                if score >= threshold
            ]
        }
        for r in records
    ]


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def evaluate(config, out, device, provenance=None, saved=lambda images: None, **detector):
    """Write `predictions.jsonl` and `summary.json` under `out`; return the summary.

    Predictions are appended image by image and `saved(images)` is called after each.
    A run interrupted before its summary continues from the images already recorded.
    `detector` holds extra arguments for the Transformers adapter.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    predictions = out / "predictions.jsonl"
    records = read_records(predictions) if predictions.exists() else []
    finished = {(r["dataset"], r["image_id"]) for r in records}
    samples = build_samples(config, out.parent / "images")
    remaining = [s for s in samples if (s["dataset"], s["image_id"]) not in finished]
    detector = make_detector(config["model"], device, **detector)
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
    score.add_argument(
        "--min-score",
        type=float,
        nargs="*",
        help="Score only boxes whose recorded confidence reaches each of these thresholds",
    )
    args = parser.parse_args()
    if args.command == "evaluate":
        summary = evaluate(json.loads(args.config.read_text()), args.out, args.device)
        print(json.dumps(summary["datasets"], indent=2))
    else:
        records = read_records(args.predictions)
        if not args.min_score:
            print(json.dumps(summarize(records), indent=2))
        else:
            sweep = {str(t): summarize(above(records, t)) for t in args.min_score}
            print(json.dumps(sweep, indent=2))


if __name__ == "__main__":
    main()
