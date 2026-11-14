"""Train and evaluate the slot head on cached Moondream 2 image features.

Two stages, both writing under one run folder:

- `encode` runs the frozen vision encoder over one shard of images, from a COCO Parquet
  shard (`coco_items`) or from images labelled by `parallel_detection.teacher`
  (`teacher_items`), and saves the 729 image embeddings per image (raw BF16 bits) with the
  image's kept objects.
- `train` loads those shards, trains the slot head and writes `metrics.jsonl`,
  `heldout.jsonl`, `head.pt`, `report.json` and a head snapshot every `snapshot_every`
  steps (default 5,000) under `snapshots/`.

Labels are filtered to the ones worth learning from: objects covering at least
`min_area` of the image, in images with at most `max_objects` of them. The task is to
find some objects reliably, not to enumerate every annotated one.

This is a single-GPU loop for a set-prediction loss. `training/trainer/loop.py` is
built around token-weighted language-model objectives and packed token data, so it does
not host this experiment.
"""

import copy
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

from parallel_detection.merge import select
from parallel_detection.slots import (
    COORDINATES,
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
            _, sampled, tokens, _ = sample(head, inner.text, inner.region, cache, rotate, steps)
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


def save_head(path, head, config, data, heldout_row):
    """Save the head with the objectness threshold that maximised held-out F1 in
    `heldout_row`; returns that threshold."""
    threshold = float(max(heldout_row["f1_iou_0.5"], key=heldout_row["f1_iou_0.5"].get))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head": head.state_dict(),
            "config": config,
            "vocabulary": data.vocabulary,
            "threshold": threshold,
        },
        path,
    )
    return threshold


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
    # Self-training: a slowly averaged copy of the head marks unmatched slots where it
    # confidently sees a COCO category; those slots get no "nothing here" gradient.
    self_ignore = config.get("self_ignore")
    teacher = copy.deepcopy(head).requires_grad_(False) if self_ignore else None
    names = torch.tensor(list(data.vocabulary.values()), device=device)
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
        if teacher is not None:
            teacher.load_state_dict(state["teacher"])
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
                        "teacher": teacher.state_dict() if teacher is not None else None,
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
                if step and step % config.get("snapshot_every", 5000) == 0:
                    # Intermediate heads stay for later study; state.pt is only for resuming.
                    save_head(
                        out / "snapshots" / f"head-step{step:05d}.pt", head, config, data, row
                    )
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
            ignore = None
            if teacher is not None and step >= self_ignore["start_step"]:
                with torch.no_grad():
                    seen, _, labels = teacher(inner.text, inner.region, cache, rotate)
                tokens = labels.argmax(-1)
                category = (tokens[:, :, None, :] == names[None, None]).all(-1).any(-1)
                ignore = (seen.sigmoid() >= self_ignore["threshold"]) & category
            terms, matching, stats = set_loss(
                objectness, box_logits, logits, targets, **options, ignore=ignore
            )
            denoise = diffusion_loss(
                head, inner.text, inner.region, cache, rotate, matching, generator
            )
            loss = total(terms) + denoise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            if teacher is not None:
                with torch.no_grad():
                    for averaged, current in zip(
                        teacher.parameters(), head.parameters(), strict=True
                    ):
                        averaged.lerp_(current, 1 - self_ignore["decay"])
            if step % config["log_every"] == 0:
                row = {"step": step, "seconds": time.time() - started, "loss": loss.item()}
                row |= {k: v.item() for k, v in terms.items()} | {"denoise": denoise.item()} | stats
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
    final = json.loads((out / "heldout.jsonl").read_text().splitlines()[-1])
    threshold = save_head(out / "head.pt", head, config, data, final)
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


# How a slot is ranked, from its objectness and the sampler's label and box confidence.
SLOT_SCORES = {
    "objectness": lambda objectness, label, box, power: objectness,
    "objectness_label_box": lambda objectness, label, box, power: objectness * label * box**power,
}


class SlotDetector:
    """The trained slot head behind the evaluation harness's detector interface.

    One pass over the full image, plus one per tile when tiling, yields every candidate;
    `merge.select` chooses each phrase's boxes. All time is reported as shared.
    """

    def __init__(
        self,
        checkpoint,
        device,
        steps=1,
        max_crops=None,
        threshold=None,
        all_slots=False,
        score="objectness_label_box",
        duplicate_iou=0.7,
        backbone=None,
        box_power=1.0,
        merge="nms",
        aliases=None,
        tiles=1,
        tile_overlap=0.25,
        tile_max_cells=None,
    ):
        """`score` ranks a slot by "objectness" alone or by "objectness_label_box", the
        product of objectness and the sampler's confidence in its label and box; a slot is
        kept when that reaches `threshold`. The checkpoint's threshold was chosen for
        objectness, so another score needs an explicit threshold. Kept boxes of the same
        label overlapping a higher-scoring one by at least `duplicate_iou` are dropped
        (None keeps them). `max_crops` overrides how many high-resolution crops the
        vision encoder reads; `all_slots` also returns every slot's box, objectness,
        label and confidences. `backbone` is the (model, inner, functions) of
        `load_backbone` to share with other detectors.

        `box_power` raises the box confidence to that power in the combined score.
        `merge` ("nms" or "fuse"), `aliases` (phrase -> head labels it covers) and
        `tile_max_cells` are passed to `merge.select`. `tiles` > 1 also runs the head on
        a `tiles` x `tiles` grid of crops overlapping by `tile_overlap` of a tile, for
        objects too small for the full image's grid."""
        if score not in SLOT_SCORES:
            raise ValueError(f"Unknown slot score {score!r}")
        if threshold is None and score != "objectness":
            raise ValueError("A score other than objectness needs an explicit threshold")
        state = torch.load(checkpoint, map_location=device)
        config = state["config"]
        self.model, self.inner, self.functions = backbone or load_backbone(
            **config["model"], device=device
        )
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
        self.all_slots, self.score, self.duplicate_iou = all_slots, score, duplicate_iou
        self.box_power, self.merge, self.aliases = box_power, merge, aliases
        self.tiles, self.tile_overlap, self.tile_max_cells = tiles, tile_overlap, tile_max_cells

    def _clock(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def views(self, width, height):
        """(name, [x0, y0, x1, y1]) of the full image and of each tile."""
        out = [("full", [0.0, 0.0, float(width), float(height)])]
        if self.tiles > 1:
            n, overlap = self.tiles, self.tile_overlap
            tile_w = width / (1 + (n - 1) * (1 - overlap))
            tile_h = height / (1 + (n - 1) * (1 - overlap))
            for row in range(n):
                for col in range(n):
                    x0, y0 = col * tile_w * (1 - overlap), row * tile_h * (1 - overlap)
                    out.append(("tile", [x0, y0, x0 + tile_w, y0 + tile_h]))
        return out

    def _slots(self, image):
        """Every slot of one view: corners in [0, 1] of that view, label, confidences."""
        embedding = self.inner._run_vision_encoder(image)
        cache = image_cache(self.inner, self.functions, embedding[None])
        objectness, boxes, tokens, confidence = sample(
            self.head,
            self.inner.text,
            self.inner.region,
            cache,
            self.functions.apply_rotary_emb,
            self.steps,
        )
        objectness = objectness[0].sigmoid()
        box_confidence = confidence[0, :, :COORDINATES].min(-1).values
        label_confidence = confidence[0, :, COORDINATES:].prod(-1)
        score = SLOT_SCORES[self.score](
            objectness, label_confidence, box_confidence, self.box_power
        ).tolist()
        names = decode_names(tokens[0], self.vocabulary)
        corners = to_corners(boxes[0]).clamp(0, 1).tolist()
        return [
            {
                "corners": corners[k],
                "label": names[k],
                "score": score[k],
                "objectness": objectness[k].item(),
                "box_confidence": box_confidence[k].item(),
                "label_confidence": label_confidence[k].item(),
            }
            for k in range(len(names))
        ]

    @torch.inference_mode()
    def detect(self, image, phrases):
        width, height = image.size
        start = self._clock()
        candidates, seconds = [], {}
        for view, (x0, y0, x1, y1) in self.views(width, height):
            began = self._clock()
            crop = image if view == "full" else image.crop((x0, y0, x1, y1))
            edges = (x0 > 0, y0 > 0, x1 < width, y1 < height)  # inner edges of this view
            for slot in self._slots(crop):
                c = slot.pop("corners")
                near = (c[0] < 0.01, c[1] < 0.01, c[2] > 0.99, c[3] > 0.99)
                slot["box"] = [
                    x0 + c[0] * (x1 - x0),
                    y0 + c[1] * (y1 - y0),
                    x0 + c[2] * (x1 - x0),
                    y0 + c[3] * (y1 - y0),
                ]
                slot["view"] = view
                slot["cut"] = any(e and n for e, n in zip(edges, near, strict=True))
                candidates.append(slot)
            seconds[view] = seconds.get(view, 0.0) + self._clock() - began
        found, scores = select(
            candidates,
            phrases,
            width,
            height,
            self.threshold,
            self.duplicate_iou,
            self.merge,
            self.aliases,
            self.tile_max_cells,
        )
        shared = self._clock() - start
        slots = {"slots": candidates} if self.all_slots else {}
        return slots | {
            "boxes": found,
            "scores": scores,
            "shared_seconds": shared,
            "phrase_seconds": [0.0] * len(phrases),
            "components": {f"{view}_views": t for view, t in seconds.items()},
        }
