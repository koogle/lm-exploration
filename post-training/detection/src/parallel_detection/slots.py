"""One-pass set detection on a frozen Moondream 2: slots filled by masked diffusion.

After the image prefix, the decoder reads `slots` blocks of 3 + `label_tokens`
positions in a single forward pass. All slot positions see the image and each other.
Every position starts as a mask and is read out through the model's own heads:

- positions 0, 1 and 2 are the box: x centre and y centre through the region head's
  coordinate decoder and width/height through its size decoder, each a distribution
  over the 1,024 bins Moondream's sequential detector uses;
- the remaining positions are the label, through the vocabulary head;
- the first position also gives an objectness logit (the slot may be "nothing").

One pass with everything masked is the one-step sampler. Further passes reveal the most
confident positions, encoded as Moondream encodes coordinates, sizes and tokens, and
predict the rest again.

The backbone weights stay frozen. Low-rank adapters act only at slot positions, so the
image prefix and every existing skill of the model are unchanged.
"""

import math

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

PREFIX = 730  # BOS + 729 image tokens
HEADS = 32


class LowRank(nn.Module):
    def __init__(self, inputs, outputs, rank):
        super().__init__()
        self.down = nn.Linear(inputs, rank, bias=False)
        self.up = nn.Linear(rank, outputs, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))


def block_forward(block, x, positions, freqs, rotate, prefix=None, adapter=None):
    """One Moondream decoder block on x [B, S, D]; returns the new x and this span's keys/values.

    `prefix` is (keys, values) of earlier positions that every position of x may attend
    to; positions of x attend to each other without a mask. The frozen weights compute
    in their own precision; x and the adapters keep the precision of x.
    """
    batch, length, dim = x.shape
    frozen = block.attn.qkv.weight.dtype
    normed = F.layer_norm(x, (dim,), block.ln.weight.to(x.dtype), block.ln.bias.to(x.dtype))
    qkv = block.attn.qkv(normed.to(frozen))
    if adapter is not None:
        qkv = (qkv + adapter["qkv"](normed)).to(frozen)
    q, k, v = (t.view(batch, length, HEADS, dim // HEADS).transpose(1, 2) for t in qkv.chunk(3, -1))
    q, k = rotate(q, freqs, positions, HEADS), rotate(k, freqs, positions, HEADS)
    keys, values = k, v
    if prefix is not None:
        keys, values = torch.cat([prefix[0], k], dim=2), torch.cat([prefix[1], v], dim=2)
    mixed = F.scaled_dot_product_attention(q, keys, values)
    mixed = mixed.transpose(1, 2).reshape(batch, length, dim)
    attended = block.attn.proj(mixed).to(x.dtype)
    hidden = block.mlp.fc1(normed.to(frozen)).to(x.dtype)
    if adapter is not None:
        attended = attended + adapter["proj"](mixed.to(x.dtype))
        hidden = hidden + adapter["fc1"](normed)
    hidden = F.gelu(hidden, approximate="tanh")
    out = block.mlp.fc2(hidden.to(frozen)).to(x.dtype)
    if adapter is not None:
        out = out + adapter["fc2"](hidden)
    return x + attended + out, (k, v)


@torch.no_grad()
def prefix_cache(text, embeddings, rotate):
    """Keys and values of the image prefix for a batch of [B, PREFIX, D] input embeddings."""
    positions = torch.arange(embeddings.size(1), device=embeddings.device)
    x, cache = embeddings, []
    for block in text.blocks:
        x, kv = block_forward(block, x, positions, text.freqs_cis, rotate)
        cache.append(kv)
    return cache


COORDINATES = 3  # x, y and size positions ahead of the label positions
BINS = 1024


def region_mlp(x, weights):
    """Moondream's two-layer region decoder, in the precision of its frozen weights."""
    x = weights.fc1(x.to(weights.fc1.weight.dtype))
    return weights.fc2(F.gelu(x, approximate="tanh")).float()


def fourier(values, features):
    """Moondream's Fourier features of values [..., n] under a frequency matrix [n, f]."""
    f = 2 * math.pi * values.to(features.dtype) @ features
    return torch.cat([f.cos(), f.sin()], dim=-1)


def bins_of(boxes):
    """Bin indices [..., 4] of boxes (cx, cy, w, h in 0..1): linear centres, log-scale sizes."""
    centre = (boxes[..., :2] * BINS).floor()
    size = ((boxes[..., 2:].clamp(1 / 1024, 1).log2() + 10) / 10 * (BINS - 1)).round()
    return torch.cat([centre, size], dim=-1).clamp(0, BINS - 1).long()


def boxes_of(bins):
    """Boxes of (possibly fractional) bin indices, as Moondream decodes them."""
    centre = bins[..., :2].float() / BINS
    size = torch.pow(2.0, bins[..., 2:].float() / (BINS - 1) * 10 - 10)
    return torch.cat([centre, size], dim=-1)


def soft_boxes(box_logits):
    """Differentiable boxes from bin logits [..., 4, BINS]: the expected bin, decoded."""
    index = torch.arange(BINS, device=box_logits.device, dtype=torch.float32)
    return boxes_of((box_logits.softmax(-1) * index).sum(-1))


class SlotHead(nn.Module):
    """The trainable parts: slot queries, the mask and role inputs, adapters and objectness."""

    def __init__(self, text, slots=32, label_tokens=3, rank=16, hints=False):
        super().__init__()
        dim, ff = text.wte.size(1), text.blocks[0].mlp.fc1.out_features
        self.slots, self.label_tokens = slots, label_tokens
        self.width = COORDINATES + label_tokens
        embeddings = text.wte.detach().float()
        scale = embeddings.std().item()
        self.queries = nn.Parameter(torch.randn(slots, dim) * scale)
        self.mask = nn.Parameter(embeddings.mean(0).clone())
        self.roles = nn.Parameter(torch.randn(self.width, dim) * scale)
        self.adapters = nn.ModuleList(
            nn.ModuleDict(
                {
                    "qkv": LowRank(dim, 3 * dim, rank),
                    "proj": LowRank(dim, dim, rank),
                    "fc1": LowRank(dim, ff, rank),
                    "fc2": LowRank(ff, dim, rank),
                }
            )
            for _ in text.blocks
        )
        self.objectness = nn.Linear(dim, 1)
        # Video: how strongly a previous frame's box and label (a hint) enter a slot.
        # Starts at zero, so a hinted head begins as the unhinted one.
        self.hint_gate = nn.Parameter(torch.zeros(())) if hints else None

    def embed(self, text, region, boxes, tokens):
        """Input embeddings [.., width, D] of boxes [.., 4] (cx, cy, w, h) and label tokens
        [.., T], through Moondream's own coordinate, size and word encoders."""
        cx = region.coord_encoder(fourier(boxes[..., 0:1], region.coord_features))
        cy = region.coord_encoder(fourier(boxes[..., 1:2], region.coord_features))
        size = region.size_encoder(fourier(boxes[..., 2:4], region.size_features))
        return torch.cat([torch.stack([cx, cy, size], dim=-2), text.wte[tokens]], dim=-2).float()

    def forward(self, text, region, cache, rotate, known=None, hints=None):
        """Return objectness logits [B, K], box bin logits [B, K, 4, BINS] (x, y, w, h)
        and label logits [B, K, T, vocabulary].

        `known` is (reveal [B, K, width] bool, bins [B, K, 4], tokens [B, K, T]): revealed
        positions get their value's embedding in place of the mask.

        `hints` is (has [B, K] bool, boxes [B, K, 4] cx, cy, w, h, tokens [B, K, T]): a
        hinted slot's positions also receive the gated embedding of that box and label,
        for example the slot's detection in the previous video frame. Unlike `known`, a
        hint is only an input: every position is still predicted.
        """
        batch = cache[0][0].size(0)
        x = (self.mask + self.roles).expand(batch, self.slots, -1, -1)
        if known is not None:
            reveal, bins, tokens = known
            given = self.embed(text, region, boxes_of(bins), tokens)
            x = torch.where(reveal.unsqueeze(-1), given + self.roles, x)
        if hints is not None:
            has, boxes, tokens = hints
            hinted = self.embed(text, region, boxes, tokens)
            x = x + self.hint_gate * hinted * has[:, :, None, None].float()
        # Every position carries its slot's query so that the block binds to one object.
        x = (x + self.queries[None, :, None, :]).flatten(1, 2)
        # Slots follow everything cached: the image prefix and any prompt.
        positions = cache[0][0].size(2) + torch.arange(x.size(1), device=x.device)
        for block, kv, adapter in zip(text.blocks, cache, self.adapters, strict=True):
            x, _ = block_forward(block, x, positions, text.freqs_cis, rotate, kv, adapter)
        hidden = x.view(batch, self.slots, self.width, -1)
        # The region decoders read the raw final hidden state, as in Moondream's detector.
        centre = region_mlp(hidden[:, :, 0:2], region.coord_decoder)
        size = region_mlp(hidden[:, :, 2], region.size_decoder).view(batch, self.slots, 2, BINS)
        normed = F.layer_norm(
            hidden, (hidden.size(-1),), text.post_ln.weight.float(), text.post_ln.bias.float()
        )
        labels = text.lm_head(normed[:, :, COORDINATES:].to(text.lm_head.weight.dtype)).float()
        objectness = self.objectness(normed[:, :, 0]).squeeze(-1)
        return objectness, torch.cat([centre, size], dim=2), labels


def to_corners(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_iou(a, b):
    """Pairwise generalized IoU of corner boxes a [N, 4] and b [M, 4]."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    inter = torch.min(a[:, None, 2:], b[None, :, 2:]) - torch.max(a[:, None, :2], b[None, :, :2])
    inter = inter.clamp(min=0).prod(-1)
    union = area_a[:, None] + area_b[None, :] - inter
    hull = torch.max(a[:, None, 2:], b[None, :, 2:]) - torch.min(a[:, None, :2], b[None, :, :2])
    hull = hull.clamp(min=0).prod(-1)
    iou = inter / union.clamp(min=1e-9)
    return iou - (hull - union) / hull.clamp(min=1e-9), iou


WEIGHTS = {"l1": 5.0, "giou": 2.0, "bins": 1.0, "objectness": 1.0, "label": 1.0, "empty": 0.1}


def match(
    objectness,
    boxes,
    log_probs,
    truth_boxes,
    truth_tokens,
    size_groups=None,
    hint_of=None,
    hint_bonus=0.0,
):
    """Hungarian matching of one image's slots to its truth objects.

    Returns slot indices and truth indices. The cost combines box distance, overlap,
    objectness and the label's negative log-likelihood. `size_groups` is (area boundary,
    small slots): the first `small slots` slots take objects covering less than the
    boundary share of the image and the other slots take the rest, unless a group has
    more objects than slots. `hint_of` [K] gives each slot's hinted truth index (-1 for
    none); matching a slot to its hinted object is `hint_bonus` cheaper, so hinted slots
    learn to keep their object.
    """
    giou, _ = generalized_iou(to_corners(boxes), to_corners(truth_boxes))
    label = -log_probs[:, None, :, :].expand(-1, truth_tokens.size(0), -1, -1)
    label = label.gather(-1, truth_tokens[None, :, :, None].expand(boxes.size(0), -1, -1, -1))
    cost = (
        WEIGHTS["l1"] * torch.cdist(boxes, truth_boxes, p=1)
        - WEIGHTS["giou"] * giou
        - WEIGHTS["objectness"] * objectness.sigmoid()[:, None]
        + WEIGHTS["label"] * label.squeeze(-1).mean(-1)
    )
    if size_groups is not None:
        boundary, small_slots = size_groups
        small_truth = truth_boxes[:, 2] * truth_boxes[:, 3] < boundary
        small_slot = torch.arange(boxes.size(0), device=boxes.device) < small_slots
        cost = cost + 1e6 * (small_slot[:, None] != small_truth[None, :])
    if hint_of is not None and truth_boxes.size(0):
        slots = torch.nonzero(hint_of >= 0).squeeze(-1)
        cost = cost.clone()
        cost[slots, hint_of[slots]] -= hint_bonus
    rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return torch.as_tensor(rows, device=boxes.device), torch.as_tensor(cols, device=boxes.device)


# An unmatched slot whose box overlaps a labelled object at least this much is a
# duplicate or a loose box on that object, not a find of an unlabelled one.
NEAR_LABEL_IOU = 0.3


def loss_options(config):
    """The `set_loss` keyword arguments a training config selects."""
    groups = config.get("size_groups")
    return {
        "size_groups": groups and (groups["boundary"], groups["small_slots"]),
        "objectness_target": config.get("objectness_target", "binary"),
        "negatives": config.get("negatives", "all"),
        "size_weight": config.get("size_weight"),
    }


def set_loss(
    objectness,
    box_logits,
    logits,
    targets,
    size_groups=None,
    objectness_target="binary",
    negatives="all",
    ignore=None,
    size_weight=None,
    hint_of=None,
    hint_bonus=0.0,
):
    """Losses for a batch; `targets` is a list of (boxes [n, 4] cxcywh, tokens [n, T]).

    Boxes are matched and regressed through their expected bins, and each bin
    distribution is also trained on the truth bin. Box and label losses apply to matched
    slots only.

    Objectness: with `objectness_target` "binary" a matched slot's target is 1; with
    "iou" it is the IoU of the slot's decoded box with its object, so confidence ranks
    box quality. Unmatched slots have target 0 with a small weight. With `negatives`
    "all" every unmatched slot gets it; with "near_labels" only unmatched slots whose box
    overlaps a labelled object (duplicates and loose boxes) do, and slots elsewhere get
    no gradient, because they may have found an object the labels omit. `ignore` [B, K]
    marks further unmatched slots to leave without gradient, for example where a
    slowly averaged copy of the head confidently sees an object the labels lack.

    `hint_of` lists each image's [K] hinted truth indices for `match` (see there).

    `size_weight` {"reference_area", "max"} weights each matched object's box, label and
    objectness terms by sqrt(reference_area / its area), clamped to [1, max], so small
    objects count more; box and label terms become weighted means.

    Returns the loss terms, the matching as (batch, slot, truth tokens, truth bins) and
    detached statistics.
    """
    boxes = soft_boxes(box_logits)
    log_probs = logits.log_softmax(-1)
    batch_index, slot_index, truth_boxes, truth_tokens = [], [], [], []
    for b, (t_boxes, t_tokens) in enumerate(targets):
        rows, cols = match(
            objectness[b],
            boxes[b],
            log_probs[b],
            t_boxes,
            t_tokens,
            size_groups,
            hint_of[b] if hint_of is not None else None,
            hint_bonus,
        )
        batch_index.append(torch.full_like(rows, b))
        slot_index.append(rows)
        truth_boxes.append(t_boxes[cols])
        truth_tokens.append(t_tokens[cols])
    batch_index, slot_index = torch.cat(batch_index), torch.cat(slot_index)
    truth_boxes, truth_tokens = torch.cat(truth_boxes), torch.cat(truth_tokens)
    truth_bins = bins_of(truth_boxes)
    predicted = boxes[batch_index, slot_index]
    matched_logits = box_logits[batch_index, slot_index]
    giou, _ = generalized_iou(to_corners(predicted), to_corners(truth_boxes))
    decoded = boxes_of(matched_logits.argmax(-1))
    _, iou = generalized_iou(to_corners(decoded), to_corners(truth_boxes))
    iou = iou.diagonal()
    matched = torch.zeros_like(objectness, dtype=torch.bool)
    matched[batch_index, slot_index] = True
    target = torch.zeros_like(objectness)
    target[batch_index, slot_index] = 1.0 if objectness_target == "binary" else iou.detach()
    weight = torch.where(matched, 1.0, WEIGHTS["empty"])
    emphasis = torch.ones_like(truth_boxes[:, 0])
    if size_weight is not None:
        area = (truth_boxes[:, 2] * truth_boxes[:, 3]).clamp(min=1e-6)
        emphasis = (size_weight["reference_area"] / area).sqrt().clamp(1.0, size_weight["max"])
        weight = weight.index_put((batch_index, slot_index), emphasis)
    if negatives == "near_labels":
        all_boxes = to_corners(boxes_of(box_logits.argmax(-1))).detach()
        for b, (t_boxes, _) in enumerate(targets):
            _, overlap = generalized_iou(all_boxes[b], to_corners(t_boxes))
            near = overlap.max(-1).values >= NEAR_LABEL_IOU
            weight[b] = torch.where(matched[b] | near, weight[b], 0.0)
    elif negatives != "all":
        raise ValueError(f"Unknown negatives setting {negatives!r}")
    if ignore is not None:
        weight = torch.where(~matched & ignore, 0.0, weight)
    if objectness_target not in ("binary", "iou"):
        raise ValueError(f"Unknown objectness target {objectness_target!r}")
    share = emphasis / emphasis.sum().clamp(min=1e-9)

    def mean(per_object):
        # A batch can have no matched object (every prompted category absent): zero.
        return (per_object * share).sum() if len(share) else per_object.sum() * 0

    terms = {
        "objectness": F.binary_cross_entropy_with_logits(objectness, target, weight),
        "l1": mean(F.l1_loss(predicted, truth_boxes, reduction="none").mean(-1)),
        "giou": mean(1 - giou.diagonal()),
        "bins": mean(
            F.cross_entropy(matched_logits.flatten(0, 1), truth_bins.flatten(), reduction="none")
            .view(-1, 4)
            .mean(-1)
        ),
        "label": mean(
            F.nll_loss(
                log_probs[batch_index, slot_index].flatten(0, 1),
                truth_tokens.flatten(),
                reduction="none",
            )
            .view(truth_tokens.shape)
            .mean(-1)
        ),
    }
    stats = {
        "matched_iou": iou.mean().item(),
        "ignored_empty_share": (
            (ignore & ~matched).sum().item() / max((~matched).sum().item(), 1)
            if ignore is not None
            else 0.0
        ),
        "supervised_empty_share": ((weight > 0) & ~matched).sum().item()
        / max((~matched).sum().item(), 1),
        "label_exact": (logits[batch_index, slot_index].argmax(-1) == truth_tokens)
        .all(-1)
        .float()
        .mean()
        .item(),
    }
    return terms, (batch_index, slot_index, truth_tokens, truth_bins), stats


def total(terms):
    return sum(WEIGHTS[name] * value for name, value in terms.items())


# Which slot position predicts each of the four box distributions (x, y, w, h).
BOX_POSITION = (0, 1, 2, 2)


def diffusion_loss(head, text, region, cache, rotate, matching, generator):
    """Masked-diffusion training step for the matched slots.

    Each matched slot gets a masking ratio drawn uniformly; revealed positions are given
    their true coordinate, size or token, and the loss is taken on the positions still
    masked.
    """
    batch_index, slot_index, truth_tokens, truth_bins = matching
    batch, device = cache[0][0].size(0), truth_tokens.device
    reveal = torch.zeros((batch, head.slots, head.width), dtype=torch.bool, device=device)
    bins = torch.zeros((batch, head.slots, 4), dtype=torch.long, device=device)
    tokens = torch.zeros((batch, head.slots, head.label_tokens), dtype=torch.long, device=device)
    count = truth_tokens.size(0)
    ratio = torch.rand(count, 1, generator=generator, device=device)
    draw = torch.rand(count, head.width, generator=generator, device=device)
    reveal[batch_index, slot_index] = draw > ratio
    bins[batch_index, slot_index] = truth_bins
    tokens[batch_index, slot_index] = truth_tokens
    _, box_logits, logits = head(text, region, cache, rotate, (reveal, bins, tokens))
    masked = ~reveal[batch_index, slot_index]
    box_masked = masked[:, list(BOX_POSITION)]
    label_masked = masked[:, COORDINATES:]
    loss = logits.sum() * 0
    if box_masked.any():
        selected = box_logits[batch_index, slot_index][box_masked]
        loss = loss + F.cross_entropy(selected, truth_bins[box_masked])
    if label_masked.any():
        selected = logits[batch_index, slot_index][label_masked]
        loss = loss + F.cross_entropy(selected, truth_tokens[label_masked])
    return loss


@torch.no_grad()
def sample(head, text, region, cache, rotate, steps):
    """Sample every slot in `steps` passes; returns objectness, boxes, label tokens and the
    confidence of each position (x, y, size, then label tokens; 1 where revealed).

    Each pass after the first reveals the most confident positions so far (an equal
    share of a slot's positions per pass) and predicts the remaining positions again.
    Objectness comes from the last pass.
    """
    known = None
    for step in range(steps):
        objectness, box_logits, logits = head(text, region, cache, rotate, known)
        box_confidence, bins = box_logits.softmax(-1).max(-1)
        token_confidence, tokens = logits.softmax(-1).max(-1)
        size_confidence = box_confidence[..., 2:].min(-1, keepdim=True).values
        confidence = torch.cat([box_confidence[..., :2], size_confidence, token_confidence], -1)
        if known is not None:
            reveal, known_bins, known_tokens = known
            bins = torch.where(reveal[..., list(BOX_POSITION)], known_bins, bins)
            tokens = torch.where(reveal[..., COORDINATES:], known_tokens, tokens)
            confidence = torch.where(reveal, torch.ones_like(confidence), confidence)
        if step + 1 < steps:
            keep = math.ceil(head.width * (step + 1) / steps)
            order = confidence.argsort(-1, descending=True)
            reveal = torch.zeros_like(confidence, dtype=torch.bool)
            known = (reveal.scatter(-1, order[..., :keep], True), bins, tokens)
    return objectness, boxes_of(bins), tokens, confidence
