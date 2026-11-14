"""Box labels from vision-language models through OpenRouter, and their consensus.

    python -m parallel_detection.teacher label --config configs/teacher-pilot.json \
        --out artifacts/teacher-pilot --budget-usd 25
    python -m parallel_detection.teacher score --out artifacts/teacher-pilot

The config names the images (`datasets`, `seed` and an optional `limit` on the first
samples) and maps each OpenRouter model id to extra request fields.

A config with `manifest` labels the unlabelled images listed there instead, at most
`per_source[source]` of each source and none answered in an `exclude_labels` file;
`score` needs `datasets`.

`label` asks every configured model for every instance of the 80 COCO categories in
each image and appends one JSON line per (model, image) to `<out>/<model>.jsonl`, with
the raw reply, parsed pixel boxes, token usage and cost. It stops before a request
would start once the recorded cost reaches the budget, and skips (model, image) pairs
already recorded. With `"tiles": n` it also asks about each tile of an n x n grid
(overlapping by `tile_overlap`, default a quarter), enlarged to the image's size so small
objects look larger, and adds the tile boxes that no tile edge cuts and that do not
repeat a box of the same label (IoU 0.5); the record keeps every tile's reply.

`score` turns each model's boxes, and their consensus, into the evaluation harness's
records for the categories COCO annotates in each image, and writes
`<out>/summary.json` with the harness's scores against COCO's labels. COCO's labels are
incomplete, so a correct box on an unlabelled object counts as a false positive there.

The key is read from the OPENROUNTER_TOKEN environment variable.
"""

import argparse
import base64
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from parallel_detection.boxes import iou_matrix
from parallel_detection.merge import add_unmatched, tile_grid
from parallel_detection.metrics import summarize
from parallel_detection.run import build_samples, fetch_image
from parallel_detection.tasks import COCO_NAMES

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
PROMPT = (
    "Detect every instance of the following object categories in the image: "
    + ", ".join(COCO_NAMES)
    + ". List every instance separately, including small, distant, partly hidden and "
    "cut-off ones. Use only these category names as labels. Answer with JSON only: "
    '{"objects": [{"label": "<category>", "box_2d": [ymin, xmin, ymax, xmax]}]}, with '
    "coordinates as integers from 0 to 1000 relative to the image height and width, "
    'origin at the top left. Answer {"objects": []} if none are present.'
)


def parse(text, width, height, order="yx"):
    """Pixel boxes [x0, y0, x1, y1] with labels from a reply; ignores unknown labels.

    `order` "yx" reads `box_2d` as [ymin, xmin, ymax, xmax], as the prompt asks; "xy"
    reads [xmin, ymin, xmax, ymax], which some models answer regardless.
    """
    match = re.search(r"\{.*\}|\[.*\]", text or "", re.DOTALL)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    items = value.get("objects", []) if isinstance(value, dict) else value
    names = set(COCO_NAMES)
    found = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        label, box = str(item.get("label", "")).strip().lower(), item.get("box_2d")
        if label not in names or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            values = [min(max(float(v), 0.0), 1000.0) for v in box]
            y0, x0, y1, x1 = values if order == "yx" else [values[i] for i in (1, 0, 3, 2)]
        except (TypeError, ValueError):
            continue
        if x1 > x0 and y1 > y0:
            box = [x0 * width / 1000, y0 * height / 1000, x1 * width / 1000, y1 * height / 1000]
            found.append({"label": label, "box": box})
    return found


def request(model, image_bytes, settings, key):
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(image_bytes).decode()
                        },
                    },
                ],
            }
        ],
        "usage": {"include": True},
    } | settings
    call = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(call, timeout=600) as response:
        return json.loads(response.read())


def images_of(config, out):
    """(samples, image folder): the evaluation tasks of `datasets`, or the unlabelled
    images of a `manifest` written by `parallel_detection.images`."""
    if "manifest" in config:
        manifest = Path(config["manifest"])
        lines = [json.loads(line) for line in manifest.read_text().splitlines()]
        # Images another labelling run already answered are skipped.
        done = {
            json.loads(line)["image_id"]
            for path in config.get("exclude_labels", [])
            for line in Path(path).read_text().splitlines()
            if "error" not in json.loads(line)
        }
        lines = [line for line in lines if line["image_id"] not in done]
        # `per_source` caps how many images of each source are labelled, in manifest order.
        quotas, kept = config.get("per_source"), []
        taken = {}
        for line in lines:
            if quotas is not None:
                if taken.get(line["source"], 0) >= quotas.get(line["source"], 0):
                    continue
                taken[line["source"]] = taken.get(line["source"], 0) + 1
            kept.append(line)
        return kept[: config.get("limit")], manifest.parent
    return build_samples(config, out / "images")[: config.get("limit")], out / "images"


def ask(model, settings, image_bytes, width, height, key):
    """One labelling request, retried on server errors, rate limits and dropped
    connections: {"text", "objects", "usage", "cost", "provider"} or {"error", "cost"}."""
    for attempt in range(3):
        try:
            reply = request(model, image_bytes, settings, key)
            message = reply["choices"][0]["message"]
            usage = reply.get("usage", {})
            return {
                "text": message.get("content"),
                "objects": parse(message.get("content"), width, height),
                "usage": usage,
                "cost": float(usage.get("cost") or 0.0),
                "provider": reply.get("provider"),
            }
        except Exception as error:  # noqa: BLE001 - every failure is recorded, not raised
            detail = error.read().decode()[:500] if hasattr(error, "read") else ""
            failed = {"error": f"{type(error).__name__}: {error} {detail}".strip(), "cost": 0.0}
            code = getattr(error, "code", None)
            if code is not None and code < 500 and code != 429:
                break
            time.sleep(5 * (attempt + 1))
    return failed


def label(config, out, budget_usd, workers=8):
    """Label the configured images with each configured model, within the budget."""
    key = os.environ["OPENROUNTER_TOKEN"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples, folder = images_of(config, out)
    lock = threading.Lock()
    spent = sum(
        json.loads(line).get("cost", 0.0)
        for path in out.glob("*.jsonl")
        for line in path.read_text().splitlines()
    )
    state = {"spent": spent}

    def one(model, settings, sample):
        path = out / (model.replace("/", "--") + ".jsonl")
        with lock:
            if state["spent"] >= budget_usd:
                return
        image = fetch_image(sample["image"], folder)
        width, height = image.size
        source = folder / sample["image"]
        image_bytes = source.read_bytes() if source.exists() else _jpeg(image)
        started = time.time()
        record = {"model": model, "image_id": sample["image_id"], "width": width, "height": height}
        record |= ask(model, settings, image_bytes, width, height, key)
        tiles = config.get("tiles", 1)
        if tiles > 1 and "error" not in record:
            # Each tile, enlarged to the image's size, so small objects look larger; boxes
            # come back in image pixels and those not cut by a tile edge are added.
            views = []
            for x0, y0, x1, y1 in tile_grid(width, height, tiles, config.get("tile_overlap", 0.25)):
                crop = image.crop((x0, y0, x1, y1)).resize((width, height))
                answer = ask(model, settings, _jpeg(crop), width, height, key)
                record["cost"] += answer["cost"]
                if "error" in answer:
                    views.append({"box": [x0, y0, x1, y1], "error": answer["error"]})
                    continue
                sx, sy = (x1 - x0) / width, (y1 - y0) / height
                inner = (x0 > 0, y0 > 0, x1 < width, y1 < height)
                found = []
                for o in answer["objects"]:
                    b = o["box"]
                    near = (
                        b[0] < 0.01 * width,
                        b[1] < 0.01 * height,
                        b[2] > 0.99 * width,
                        b[3] > 0.99 * height,
                    )
                    if any(i and n for i, n in zip(inner, near, strict=True)):
                        continue
                    box = [x0 + b[0] * sx, y0 + b[1] * sy, x0 + b[2] * sx, y0 + b[3] * sy]
                    found.append({"label": o["label"], "box": box})
                views.append({"box": [x0, y0, x1, y1], "objects": found, "text": answer["text"]})
                record["objects"] = add_unmatched(record["objects"], found, 0.5)
            record["tiles"] = views
        record["seconds"] = time.time() - started
        with lock:
            state["spent"] += record["cost"]
            with path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

    jobs = []
    workers = config.get("workers", workers)
    for model, settings in config["models"].items():
        path = out / (model.replace("/", "--") + ".jsonl")
        done = set()
        if path.exists():
            done = {
                json.loads(line)["image_id"]
                for line in path.read_text().splitlines()
                if "error" not in json.loads(line)
            }
        jobs += [(model, settings, s) for s in samples if s["image_id"] not in done]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda job: one(*job), jobs))
    return state["spent"]


def _jpeg(image):
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def consensus(by_model, votes, threshold=0.5):
    """Boxes that at least `votes` models agree on: same label, IoU >= threshold.

    `by_model` maps model -> list of {"label", "box"}. Each kept box is the mean of the
    agreeing boxes, at most one per model.
    """
    pool = [(model, obj) for model, objects in by_model.items() for obj in objects]
    used = [False] * len(pool)
    kept = []
    for i, (model, obj) in enumerate(pool):
        if used[i]:
            continue
        group, models = [obj["box"]], {model}
        used[i] = True
        for j in range(i + 1, len(pool)):
            other_model, other = pool[j]
            if used[j] or other_model in models or other["label"] != obj["label"]:
                continue
            if iou_matrix([obj["box"]], [other["box"]])[0, 0] >= threshold:
                group.append(other["box"])
                models.add(other_model)
                used[j] = True
        if len(models) >= votes:
            mean = [sum(values) / len(values) for values in zip(*group, strict=True)]
            kept.append({"label": obj["label"], "box": mean})
    return kept


def score(config, out):
    """Score each model and the consensus against COCO's labels; write summary.json."""
    out = Path(out)
    samples = {
        s["image_id"]: s for s in build_samples(config, out / "images")[: config.get("limit")]
    }
    replies = defaultdict(dict)  # image_id -> source -> objects
    usage = {}
    # Models that may ignore the requested coordinate order are scored both ways.
    both_orders = config.get("score_both_orders", [])
    for path in sorted(out.glob("*.jsonl")):
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        ok = [r for r in lines if "error" not in r]
        model = ok[0]["model"] if ok else path.stem
        for r in ok:
            replies[r["image_id"]][model] = parse(r["text"], r["width"], r["height"])
            if model in both_orders:
                replies[r["image_id"]][f"{model} (x first)"] = parse(
                    r["text"], r["width"], r["height"], order="xy"
                )
        usage[model] = {
            "replies": len(ok),
            "errors": len(lines) - len(ok),
            "cost_usd": sum(r.get("cost", 0.0) for r in lines),
            "mean_completion_tokens": sum(
                r.get("usage", {}).get("completion_tokens", 0) for r in ok
            )
            / max(len(ok), 1),
            "mean_seconds": sum(r["seconds"] for r in ok) / max(len(ok), 1),
        }
    models = sorted(usage)
    # Only images every model answered, so all rows are scored on the same images.
    common = [i for i in samples if all(m in replies[i] for m in models)]
    names = sorted({name for i in common for name in replies[i]})
    sources = {name: (lambda i, n=name: replies[i][n]) for name in names}
    voters = config.get("consensus_models", models)
    for votes in config.get("consensus_votes", [2]):
        sources[f"consensus-{votes}"] = lambda i, v=votes: consensus(
            {m: replies[i][m] for m in voters}, v
        )
    records = []
    for name, objects_of in sources.items():
        for image_id in common:
            objects = objects_of(image_id)
            for query in samples[image_id]["queries"]:
                predicted = [o["box"] for o in objects if o["label"] == query["phrase"]]
                records.append(
                    query
                    | {
                        "dataset": name,
                        "image_id": image_id,
                        "predicted": predicted,
                        "seconds": 0.0,
                        "shared_seconds": 0.0,
                    }
                )
    summary = {"images": len(common), "usage": usage, "scores": summarize(records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("label")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--budget-usd", type=float, required=True)
    rate = commands.add_parser("score")
    rate.add_argument("--config", type=Path, required=True)
    rate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command == "label":
        print(json.dumps({"spent_usd": label(config, args.out, args.budget_usd)}))
    else:
        print(json.dumps(score(config, args.out)["usage"], indent=2))


if __name__ == "__main__":
    main()
