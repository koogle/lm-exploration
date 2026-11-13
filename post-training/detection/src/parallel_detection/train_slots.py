"""Train and evaluate the slot head on cached Moondream 2 image features.

Two stages, both writing under one run folder:

- `encode` runs the frozen vision encoder over one shard of images, from a COCO Parquet
  shard (`coco_items`) or from images labelled by `parallel_detection.teacher`
  (`teacher_items`), and saves the 729 image embeddings per image (raw BF16 bits) with the
  image's kept objects.
- `train` loads those shards, trains the slot head and writes `metrics.jsonl`,
  `heldout.jsonl`, `head.pt` and `report.json`.

Labels are filtered to the ones worth learning from: objects covering at least
`min_area` of the image, in images with at most `max_objects` of them. The task is to
find some objects reliably, not to enumerate every annotated one.

This is a single-GPU loop for a set-prediction loss. `training/trainer/loop.py` is
built around token-weighted language-model objectives and packed token data, so it does
not host this experiment.
"""

import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from parallel_detection.slots import (
    PREFIX,
    SlotHead,
    boxes_of,
    diffusion_loss,
    generalized_iou,
    loss_options,
    prefix_cache,
    sample,
    set_loss,
    to_corners,
    total,
)
from parallel_detection.tasks import COCO_NAMES

IMAGE_TOKENS = PREFIX - 1
# Objects covering less than this share of the image are reported as small.
SMALL_AREA = 0.01


def load_backbone(repo, revision, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map={"": device},
    ).eval()
    model.requires_grad_(False)
    inner = model.model
    # The model's Hub code: its text functions sit next to the model class, rope beside it.
    module = type(inner).__module__
    functions = SimpleNamespace(
        text_encoder=sys.modules[module].text_encoder,
        apply_rotary_emb=sys.modules[module.rsplit(".", 1)[0] + ".rope"].apply_rotary_emb,
    )
    return model, inner, functions


def kept_objects(objects, width, height, min_area, names=COCO_NAMES):
    """Objects covering at least `min_area` of the image, as (name, [cx, cy, w, h] in 0..1)."""
    kept = []
    for category, (x0, y0, x1, y1) in zip(objects["category"], objects["bbox"], strict=True):
        w, h = (x1 - x0) / width, (y1 - y0) / height
        if w * h >= min_area:
            kept.append((names[category], [(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, w, h]))
    return kept


def encode(items, out, config, device="cuda"):
    """Write `<out>.npy` ([N, 729, D] BF16 bits as int16) and `<out>.json` (objects).

    `items` yields (image bytes, record) with records already filtered to usable images.
    """
    import io

    from PIL import Image

    items = list(items)
    _, inner, _ = load_backbone(**config["model"], device=device)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(
        out.with_suffix(".npy"), mode="w+", dtype=np.int16, shape=(len(items), IMAGE_TOKENS, 2048)
    )
    order = []
    for index, (data, record) in enumerate(items):
        image = Image.open(io.BytesIO(data() if callable(data) else data)).convert("RGB")
        with torch.inference_mode():
            embedding = inner._run_vision_encoder(image)
        features[index] = embedding.to(torch.bfloat16).view(torch.int16).cpu().numpy()
        order.append(record)
    features.flush()
    out.with_suffix(".json").write_text(json.dumps(order))
    return len(order)


def usable(objects, config):
    return 1 <= len(objects) <= config["max_objects"]


def coco_items(parquet, config):
    """(image bytes, record) for the usable images of a detection-datasets/coco shard."""
    import pyarrow.parquet as pq

    table = pq.ParquetFile(parquet)
    records = {}
    for group in range(table.metadata.num_row_groups):
        columns = ["image_id", "width", "height", "objects"]
        for row in table.read_row_group(group, columns=columns).to_pylist():
            objects = kept_objects(row["objects"], row["width"], row["height"], config["min_area"])
            if usable(objects, config):
                records[row["image_id"]] = {"image_id": row["image_id"], "objects": objects}
    for group in range(table.metadata.num_row_groups):
        for row in table.read_row_group(group, columns=["image_id", "image"]).to_pylist():
            if row["image_id"] in records:
                yield row["image"]["bytes"], records[row["image_id"]]


def teacher_items(labels, images, config):
    """(image reader, record) for usable images labelled by `parallel_detection.teacher`.

    `labels` is the teacher's JSONL file and `images` the folder its manifest paths are
    relative to. Boxes are in pixels of the stored image.
    """
    for line in Path(labels).read_text().splitlines():
        reply = json.loads(line)
        if "error" in reply:
            continue
        objects = {
            "category": [COCO_NAMES.index(o["label"]) for o in reply["objects"]],
            "bbox": [o["box"] for o in reply["objects"]],
        }
        kept = kept_objects(objects, reply["width"], reply["height"], config["min_area"])
        if usable(kept, config):
            path = Path(images) / f"{reply['image_id']}.jpg"
            yield path.read_bytes, {"image_id": reply["image_id"], "objects": kept}


class Features:
    """Memory-mapped encoded shards, with labels tokenized into fixed-length blocks."""

    def __init__(self, stems, tokenizer, label_tokens, pad):
        self.arrays = [np.load(Path(stem).with_suffix(".npy"), mmap_mode="r") for stem in stems]
        self.records = [
            r for stem in stems for r in json.loads(Path(stem).with_suffix(".json").read_text())
        ]
        self.index = [(a, i) for a, array in enumerate(self.arrays) for i in range(len(array))]
        self.vocabulary = {}
        for name in COCO_NAMES:
            ids = tokenizer.encode(" " + name).ids
            if len(ids) > label_tokens:
                raise ValueError(f"{name!r} needs {len(ids)} label tokens")
            self.vocabulary[name] = ids + [pad] * (label_tokens - len(ids))

    def __len__(self):
        return len(self.index)

    def batch(self, indices, device):
        arrays = [self.arrays[self.index[i][0]][self.index[i][1]] for i in indices]
        embeddings = torch.from_numpy(np.stack(arrays)).to(device).view(torch.bfloat16)
        targets = []
        for i in indices:
            objects = self.records[i]["objects"]
            boxes = torch.tensor([box for _, box in objects], device=device, dtype=torch.float32)
            tokens = torch.tensor([self.vocabulary[name] for name, _ in objects], device=device)
            targets.append((boxes, tokens))
        return embeddings, targets


def image_cache(inner, functions, embeddings):
    """Prefix keys and values for a batch of image embeddings [B, 729, D]."""
    bos = functions.text_encoder(
        torch.tensor([[inner.config.tokenizer.bos_id]], device=embeddings.device), inner.text
    )
    inputs = torch.cat([bos.expand(embeddings.size(0), -1, -1), embeddings.to(bos.dtype)], dim=1)
    return prefix_cache(inner.text, inputs, functions.apply_rotary_emb)


def decode_names(tokens, vocabulary):
    """Map label token blocks [.., T] to category names; None where no category matches."""
    names = {tuple(ids): name for name, ids in vocabulary.items()}
    flat = tokens.reshape(-1, tokens.size(-1)).tolist()
    return [names.get(tuple(ids)) for ids in flat]


@torch.no_grad()
def heldout_scores(head, inner, functions, data, indices, batch_size, device, thresholds, options):
    """Loss terms, box and label quality by sampler passes, and one-pass F1 at IoU 0.5 per
    objectness threshold.

    A detection counts when its box overlaps an unmatched truth box of the same decoded
    label by at least 0.5.
    """
    from parallel_detection.boxes import match as match_boxes

    head.eval()
    sums, counts = {}, 0
    tallies = {t: [0, 0, 0] for t in thresholds}  # tp, fp, fn
    by_size = {t: {"small": [0, 0], "large": [0, 0]} for t in thresholds}  # found, total
    undecodable = decoded = 0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        embeddings, targets = data.batch(chunk, device)
        cache = image_cache(inner, functions, embeddings)
        rotate = functions.apply_rotary_emb
        objectness, box_logits, logits = head(inner.text, inner.region, cache, rotate)
        terms, (b_index, s_index, truth_tokens, truth_bins), stats = set_loss(
            objectness, box_logits, logits, targets, **options
        )
        values = {k: v.item() for k, v in terms.items()} | stats
        truth = to_corners(boxes_of(truth_bins))
        for steps in (2, 3):
            _, sampled, tokens = sample(head, inner.text, inner.region, cache, rotate, steps)
            _, iou = generalized_iou(to_corners(sampled[b_index, s_index]), truth)
            values[f"matched_iou_{steps}_steps"] = iou.diagonal().mean().item()
            values[f"label_exact_{steps}_steps"] = (
                (tokens[b_index, s_index] == truth_tokens).all(-1).float().mean().item()
            )
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + value * len(chunk)
        counts += len(chunk)
        names = decode_names(logits.argmax(-1), data.vocabulary)
        scores = objectness.sigmoid()
        corners = to_corners(boxes_of(box_logits.argmax(-1))).cpu().numpy()
        for b, i in enumerate(chunk):
            truth = data.records[i]["objects"]
            for threshold in thresholds:
                keep = [k for k in range(head.slots) if scores[b, k] >= threshold]
                decoded += len(keep) * (threshold == thresholds[0])
                undecodable += sum(names[b * head.slots + k] is None for k in keep) * (
                    threshold == thresholds[0]
                )
                for name in {n for n, _ in truth} | {names[b * head.slots + k] for k in keep}:
                    predicted = [corners[b][k] for k in keep if names[b * head.slots + k] == name]
                    wanted = [box for n, box in truth if n == name]
                    pairs = (
                        match_boxes(
                            predicted,
                            [to_corners(torch.tensor(box)).numpy() for box in wanted],
                            0.5,
                        )
                        if predicted and wanted
                        else []
                    )
                    tallies[threshold][0] += len(pairs)
                    tallies[threshold][1] += len(predicted) - len(pairs)
                    tallies[threshold][2] += len(wanted) - len(pairs)
                    found = {t for _, t in pairs}
                    for index, box in enumerate(wanted):
                        size = "small" if box[2] * box[3] < SMALL_AREA else "large"
                        by_size[threshold][size][0] += index in found
                        by_size[threshold][size][1] += 1
    head.train()
    out = {key: value / counts for key, value in sums.items()}
    out["undecodable_label_rate"] = undecodable / max(decoded, 1)
    out["f1_iou_0.5"] = {
        str(t): 2 * tp / max(2 * tp + fp + fn, 1) for t, (tp, fp, fn) in tallies.items()
    }
    out["precision_iou_0.5"] = {str(t): tp / max(tp + fp, 1) for t, (tp, fp, _) in tallies.items()}
    out["recall_iou_0.5"] = {str(t): tp / max(tp + fn, 1) for t, (tp, _, fn) in tallies.items()}
    out["recall_by_size_iou_0.5"] = {
        str(t): {size: found / max(count, 1) for size, (found, count) in sizes.items()}
        for t, sizes in by_size.items()
    }
    return out


def train(config, features, out, device="cuda", saved=lambda: None, deadline=None):
    """Train the slot head; `features` lists encoded shard stems. Returns the report."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    _, inner, functions = load_backbone(**config["model"], device=device)
    tokens = inner.config.tokenizer
    data = Features(features, inner.tokenizer, config["label_tokens"], tokens.eos_id)
    order = torch.randperm(
        len(data), generator=torch.Generator().manual_seed(config["seed"])
    ).tolist()
    heldout, training = order[: config["heldout_images"]], order[config["heldout_images"] :]
    head = SlotHead(inner.text, config["slots"], config["label_tokens"], config["rank"]).to(device)
    adapters = list(head.adapters.parameters())
    others = [p for name, p in head.named_parameters() if not name.startswith("adapters.")]
    groups = [
        {"params": adapters, "base": config["learning_rate"]},
        {"params": others, "base": config["head_learning_rate"]},
    ]
    optimizer = torch.optim.AdamW(groups, lr=config["learning_rate"], weight_decay=0.01)
    sampler = random.Random(config["seed"])
    steps, warmup = config["steps"], config["warmup_steps"]

    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1)))

    options = loss_options(config)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    first_step, elapsed = 0, 0.0
    state_path = out / "state.pt"
    if state_path.exists():
        # A preempted run continues from its last evaluation boundary.
        # On CPU: generator states must be CPU byte tensors even for a CUDA generator, and
        # load_state_dict moves the weights and optimizer state to the parameters' device.
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.setstate(state["sampler"])
        generator.set_state(state["generator"])
        first_step, elapsed = state["step"], state["seconds"]
        for name in ("metrics.jsonl", "heldout.jsonl"):
            kept = [r for r in map(json.loads, (out / name).read_text().splitlines())]
            kept = [r for r in kept if r["step"] < first_step]
            (out / name).write_text("".join(json.dumps(r) + "\n" for r in kept))
    started = time.time() - elapsed
    stop_reason = "completed"
    with (out / "metrics.jsonl").open("a") as metrics, (out / "heldout.jsonl").open("a") as scores:
        for step in range(first_step, steps + 1):
            if step % config["evaluate_every"] == 0 or step == steps:
                torch.save(
                    {
                        "head": head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sampler": sampler.getstate(),
                        "generator": generator.get_state(),
                        "step": step,
                        "seconds": time.time() - started,
                    },
                    state_path,
                )
                row = heldout_scores(
                    head,
                    inner,
                    functions,
                    data,
                    heldout,
                    config["batch_size"],
                    device,
                    thresholds,
                    options,
                )
                scores.write(
                    json.dumps({"step": step, "seconds": time.time() - started} | row) + "\n"
                )
                scores.flush()
                saved()
            if step == steps:
                break
            if deadline is not None and time.time() > deadline:
                stop_reason = "deadline"
                break
            for group in optimizer.param_groups:
                group["lr"] = group["base"] * rate(step)
            picks = sampler.sample(training, config["batch_size"])
            embeddings, targets = data.batch(picks, device)
            cache = image_cache(inner, functions, embeddings)
            rotate = functions.apply_rotary_emb
            objectness, box_logits, logits = head(inner.text, inner.region, cache, rotate)
            terms, matching, stats = set_loss(objectness, box_logits, logits, targets, **options)
            denoise = diffusion_loss(
                head, inner.text, inner.region, cache, rotate, matching, generator
            )
            loss = total(terms) + denoise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            if step % config["log_every"] == 0:
                row = {"step": step, "seconds": time.time() - started, "loss": loss.item()}
                row |= {k: v.item() for k, v in terms.items()} | {"denoise": denoise.item()} | stats
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
    final = json.loads((out / "heldout.jsonl").read_text().splitlines()[-1])
    threshold = max(final["f1_iou_0.5"], key=final["f1_iou_0.5"].get)
    torch.save(
        {
            "head": head.state_dict(),
            "config": config,
            "vocabulary": data.vocabulary,
            "threshold": float(threshold),
        },
        out / "head.pt",
    )
    report = {
        "config": config,
        "stop_reason": stop_reason,
        "training_images": len(training),
        "heldout_images": len(heldout),
        "trainable_parameters": sum(p.numel() for p in head.parameters()),
        "train_seconds": time.time() - started,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "threshold": float(threshold),
        "final_heldout": final,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    state_path.unlink()
    saved()
    return report


class SlotDetector:
    """The trained slot head behind the evaluation harness's detector interface.

    One pass yields every detection; a phrase gets the slots above the objectness
    threshold whose decoded label equals it. All time is reported as shared.
    """

    def __init__(self, checkpoint, device, steps=1, max_crops=None, threshold=None):
        """`max_crops` overrides how many high-resolution crops the vision encoder reads;
        `threshold` overrides the objectness threshold chosen on held-out images."""
        state = torch.load(checkpoint, map_location=device)
        config = state["config"]
        self.model, self.inner, self.functions = load_backbone(**config["model"], device=device)
        if max_crops is not None:
            vision = replace(self.inner.config.vision, max_crops=max_crops)
            self.inner.config = replace(self.inner.config, vision=vision)
        self.head = SlotHead(
            self.inner.text, config["slots"], config["label_tokens"], config["rank"]
        )
        self.head.load_state_dict(state["head"])
        self.head.to(device).eval()
        self.vocabulary = state["vocabulary"]
        self.threshold = state["threshold"] if threshold is None else threshold
        self.device, self.steps = torch.device(device), steps

    def _clock(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    @torch.inference_mode()
    def detect(self, image, phrases):
        width, height = image.size
        start = self._clock()
        embedding = self.inner._run_vision_encoder(image)
        encoded = self._clock()
        cache = image_cache(self.inner, self.functions, embedding[None])
        prefilled = self._clock()
        objectness, boxes, tokens = sample(
            self.head,
            self.inner.text,
            self.inner.region,
            cache,
            self.functions.apply_rotary_emb,
            self.steps,
        )
        scores = objectness[0].sigmoid().tolist()
        keep = [score >= self.threshold for score in scores]
        names = decode_names(tokens[0], self.vocabulary)
        corners = to_corners(boxes[0]).clamp(0, 1).tolist()
        shared = self._clock() - start
        scale = [width, height, width, height]
        found = [
            [
                [c * s for c, s in zip(corners[k], scale, strict=True)]
                for k in range(len(keep))
                if keep[k] and names[k] == phrase
            ]
            for phrase in phrases
        ]
        kept = [
            [scores[k] for k in range(len(keep)) if keep[k] and names[k] == phrase]
            for phrase in phrases
        ]
        return {
            "boxes": found,
            "scores": kept,
            "shared_seconds": shared,
            "phrase_seconds": [0.0] * len(phrases),
            "components": {
                "vision_encoder": encoded - start,
                "image_prefix": prefilled - encoded,
                "slots": shared - (prefilled - start),
            },
        }
