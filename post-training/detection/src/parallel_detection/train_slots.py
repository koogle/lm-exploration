"""Train and evaluate the slot head on cached Moondream 2 image features.

Two stages, both writing under one run folder:

- `encode_shard` runs the frozen vision encoder over one COCO Parquet shard and saves
  the 729 image embeddings per image (raw BF16 bits) with the image's kept objects.
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
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from parallel_detection.slots import (
    PREFIX,
    SlotHead,
    diffusion_loss,
    prefix_cache,
    sample_labels,
    set_loss,
    to_corners,
    total,
)
from parallel_detection.tasks import COCO_NAMES

IMAGE_TOKENS = PREFIX - 1


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


def encode_shard(parquet, out, config, device="cuda"):
    """Write `<out>.npy` ([N, 729, D] BF16 bits as int16) and `<out>.json` (objects)."""
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    _, inner, _ = load_backbone(**config["model"], device=device)
    table = pq.ParquetFile(parquet)
    columns = ["image_id", "width", "height", "objects"]
    rows = [
        r
        for g in range(table.metadata.num_row_groups)
        for r in table.read_row_group(g, columns=columns).to_pylist()
    ]
    records = {}
    for row in rows:
        objects = kept_objects(row["objects"], row["width"], row["height"], config["min_area"])
        if 1 <= len(objects) <= config["max_objects"]:
            records[row["image_id"]] = {"image_id": row["image_id"], "objects": objects}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(
        out.with_suffix(".npy"), mode="w+", dtype=np.int16, shape=(len(records), IMAGE_TOKENS, 2048)
    )
    order = []
    for group in range(table.metadata.num_row_groups):
        batch = table.read_row_group(group, columns=["image_id", "image"]).to_pylist()
        for row in batch:
            if row["image_id"] not in records:
                continue
            image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            with torch.inference_mode():
                embedding = inner._run_vision_encoder(image)
            features[len(order)] = embedding.to(torch.bfloat16).view(torch.int16).cpu().numpy()
            order.append(records[row["image_id"]])
    features.flush()
    out.with_suffix(".json").write_text(json.dumps(order))
    return len(order)


class Features:
    """Encoded shards in memory, with labels tokenized into fixed-length blocks."""

    def __init__(self, stems, tokenizer, label_tokens, pad):
        self.arrays = [np.load(Path(stem).with_suffix(".npy")) for stem in stems]
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
def heldout_scores(head, inner, functions, data, indices, batch_size, device, thresholds):
    """Loss terms, label accuracy by sampler steps and F1 at IoU 0.5 per objectness threshold.

    A detection counts when its box overlaps an unmatched truth box of the same decoded
    label by at least 0.5.
    """
    from parallel_detection.boxes import match as match_boxes

    head.eval()
    sums, counts = {}, 0
    tallies = {t: [0, 0, 0] for t in thresholds}  # tp, fp, fn
    undecodable = decoded = 0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        embeddings, targets = data.batch(chunk, device)
        cache = image_cache(inner, functions, embeddings)
        objectness, boxes, logits = head(inner.text, cache, functions.apply_rotary_emb)
        terms, (b_index, s_index, truth_tokens), stats = set_loss(
            objectness, boxes, logits, targets
        )
        values = {k: v.item() for k, v in terms.items()} | stats
        _, _, two_step, _ = sample_labels(head, inner.text, cache, functions.apply_rotary_emb, 2)
        values["label_exact_2_steps"] = (
            (two_step[b_index, s_index] == truth_tokens).all(-1).float().mean().item()
        )
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + value * len(chunk)
        counts += len(chunk)
        names = decode_names(logits.argmax(-1), data.vocabulary)
        scores = objectness.sigmoid()
        corners = to_corners(boxes).cpu().numpy()
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
                    wanted = [
                        to_corners(torch.tensor(box)).numpy() for n, box in truth if n == name
                    ]
                    found = len(match_boxes(predicted, wanted, 0.5)) if predicted and wanted else 0
                    tallies[threshold][0] += found
                    tallies[threshold][1] += len(predicted) - found
                    tallies[threshold][2] += len(wanted) - found
    head.train()
    out = {key: value / counts for key, value in sums.items()}
    out["undecodable_label_rate"] = undecodable / max(decoded, 1)
    out["f1_iou_0.5"] = {
        str(t): 2 * tp / max(2 * tp + fp + fn, 1) for t, (tp, fp, fn) in tallies.items()
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
    steps, warmup = config["steps"], config["warmup_steps"]

    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1)))

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    started = time.time()
    stop_reason = "completed"
    with (out / "metrics.jsonl").open("w") as metrics, (out / "heldout.jsonl").open("w") as scores:
        for step in range(steps + 1):
            if step % config["evaluate_every"] == 0 or step == steps:
                row = heldout_scores(
                    head, inner, functions, data, heldout, config["batch_size"], device, thresholds
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
            picks = [
                training[(step * config["batch_size"] + j) % len(training)]
                for j in range(config["batch_size"])
            ]
            embeddings, targets = data.batch(picks, device)
            cache = image_cache(inner, functions, embeddings)
            objectness, boxes, logits = head(inner.text, cache, functions.apply_rotary_emb)
            terms, matching, stats = set_loss(objectness, boxes, logits, targets)
            denoise = diffusion_loss(
                head, inner.text, cache, functions.apply_rotary_emb, matching, generator
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
    saved()
    return report


class SlotDetector:
    """The trained slot head behind the evaluation harness's detector interface.

    One pass yields every detection; a phrase gets the slots above the objectness
    threshold whose decoded label equals it. All time is reported as shared.
    """

    def __init__(self, checkpoint, device, steps=1):
        state = torch.load(checkpoint, map_location=device)
        config = state["config"]
        self.model, self.inner, self.functions = load_backbone(**config["model"], device=device)
        self.head = SlotHead(
            self.inner.text, config["slots"], config["label_tokens"], config["rank"]
        )
        self.head.load_state_dict(state["head"])
        self.head.to(device).eval()
        self.vocabulary, self.threshold = state["vocabulary"], state["threshold"]
        self.device, self.steps = torch.device(device), steps

    def _clock(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def detect(self, image, phrases):
        width, height = image.size
        start = self._clock()
        encoded = self.model.encode_image(image)
        objectness, boxes, tokens, _ = sample_labels(
            self.head, self.inner.text, encoded.caches, self.functions.apply_rotary_emb, self.steps
        )
        keep = (objectness[0].sigmoid() >= self.threshold).tolist()
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
        return {"boxes": found, "shared_seconds": shared, "phrase_seconds": [0.0] * len(phrases)}
