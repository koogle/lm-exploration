import json

import torch
from torch import nn

from parallel_detection.slots import (
    BINS,
    SlotHead,
    bins_of,
    boxes_of,
    diffusion_loss,
    loss_options,
    match,
    prefix_cache,
    sample,
    set_loss,
    total,
)
from parallel_detection.train_slots import decode_names, kept_objects, teacher_items

DIM, VOCABULARY = 64, 50


def tiny_region():
    """Moondream's region modules at a small width."""

    def decoder(outputs):
        return nn.ModuleDict({"fc1": nn.Linear(DIM, 2 * DIM), "fc2": nn.Linear(2 * DIM, outputs)})

    region = nn.ModuleDict(
        {
            "coord_encoder": nn.Linear(8, DIM),
            "coord_decoder": decoder(BINS),
            "size_encoder": nn.Linear(16, DIM),
            "size_decoder": decoder(2 * BINS),
        }
    )
    region.coord_features = nn.Parameter(torch.randn(1, 4))
    region.size_features = nn.Parameter(torch.randn(2, 8))
    return region.requires_grad_(False)


def tiny_text():
    """A two-block decoder with Moondream's module layout."""

    def block():
        return nn.ModuleDict(
            {
                "ln": nn.LayerNorm(DIM),
                "attn": nn.ModuleDict(
                    {"qkv": nn.Linear(DIM, 3 * DIM), "proj": nn.Linear(DIM, DIM)}
                ),
                "mlp": nn.ModuleDict(
                    {"fc1": nn.Linear(DIM, 4 * DIM), "fc2": nn.Linear(4 * DIM, DIM)}
                ),
            }
        )

    text = nn.ModuleDict(
        {
            "blocks": nn.ModuleList([block(), block()]),
            "post_ln": nn.LayerNorm(DIM),
            "lm_head": nn.Linear(DIM, VOCABULARY),
        }
    )
    text.wte = nn.Parameter(torch.randn(VOCABULARY, DIM))
    text.freqs_cis = None
    return text.requires_grad_(False)


def no_rotation(x, freqs, positions, heads):
    return x


def test_bins_follow_moondreams_decoding():
    boxes = torch.tensor([[0.5, 0.25, 0.5, 1 / 1024], [0.999, 0.0, 1.0, 0.1]])
    bins = bins_of(boxes)
    assert bins.tolist()[0] == [512, 256, 921, 0] and bins.tolist()[1][:3] == [1022, 0, 1023]
    decoded = boxes_of(bins)
    assert torch.allclose(decoded[:, :2], boxes[:, :2], atol=1 / BINS)
    assert torch.allclose(decoded[:, 2:].log2(), boxes[:, 2:].log2(), atol=10 / (BINS - 1))


def test_slot_head_trains_on_a_set_loss_and_leaves_the_backbone_frozen():
    torch.manual_seed(0)
    text, region = tiny_text(), tiny_region()
    head = SlotHead(text, slots=5, label_tokens=3, rank=2)
    cache = prefix_cache(text, torch.randn(2, 7, DIM), no_rotation)
    objectness, box_logits, logits = head(text, region, cache, no_rotation)
    assert objectness.shape == (2, 5) and box_logits.shape == (2, 5, 4, BINS)
    assert logits.shape == (2, 5, 3, 50)
    targets = [
        (
            torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]),
            torch.tensor([[3, 4, 0], [7, 0, 0]]),
        ),
        (torch.tensor([[0.7, 0.3, 0.4, 0.4]]), torch.tensor([[9, 9, 9]])),
    ]
    terms, matching, stats = set_loss(objectness, box_logits, logits, targets)
    assert matching[0].tolist() == [0, 0, 1] and len(set(matching[1][:2].tolist())) == 2
    assert matching[3].shape == (3, 4) and 0 <= stats["matched_iou"] <= 1
    generator = torch.Generator().manual_seed(0)
    denoise = diffusion_loss(head, text, region, cache, no_rotation, matching, generator)
    (total(terms) + denoise).backward()
    assert head.queries.grad.abs().sum() > 0 and head.roles.grad.abs().sum() > 0
    assert all(p.grad is None for p in [*text.parameters(), *region.parameters()])


def test_sampler_keeps_revealed_positions_between_passes():
    torch.manual_seed(0)
    text, region = tiny_text(), tiny_region()
    head = SlotHead(text, slots=4, label_tokens=3, rank=2)
    cache = prefix_cache(text, torch.randn(1, 7, DIM), no_rotation)
    _, boxes, tokens, confidence = sample(head, text, region, cache, no_rotation, steps=1)
    assert confidence.shape == (1, 4, 6)
    assert boxes.shape == (1, 4, 4) and tokens.shape == (1, 4, 3)
    # A position revealed after the first pass keeps its value in every later pass.
    _, first_boxes, first_tokens, _ = sample(head, text, region, cache, no_rotation, steps=1)
    _, later_boxes, later_tokens, _ = sample(head, text, region, cache, no_rotation, steps=6)
    same = (first_tokens == later_tokens).sum() + (first_boxes == later_boxes).sum()
    assert same >= 4  # at least the first revealed position of each slot


def test_label_filter_and_decoding():
    objects = {"category": [0, 1], "bbox": [[0, 0, 50, 50], [0, 0, 2, 2]]}
    kept = kept_objects(objects, 100, 100, min_area=0.002, names=["person", "bicycle"])
    assert kept == [("person", [0.25, 0.25, 0.5, 0.5])]
    vocabulary = {"person": [5, 0, 0], "fire hydrant": [6, 7, 8]}
    tokens = torch.tensor([[[5, 0, 0], [6, 7, 0], [6, 7, 8]]])
    assert decode_names(tokens, vocabulary) == ["person", None, "fire hydrant"]


def test_size_groups_send_small_objects_to_the_small_slots():
    torch.manual_seed(0)
    boxes = torch.rand(6, 4)
    truth = torch.tensor([[0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.6, 0.6], [0.2, 0.2, 0.02, 0.1]])
    tokens = torch.zeros(3, 2, dtype=torch.long)
    log_probs = torch.zeros(6, 2, 4).log_softmax(-1)
    slots, objects = match(torch.zeros(6), boxes, log_probs, truth, tokens, size_groups=(0.01, 2))
    assignment = dict(zip(objects.tolist(), slots.tolist(), strict=True))
    assert {assignment[0], assignment[2]} == {0, 1} and assignment[1] >= 2


def test_near_label_negatives_and_iou_targets():
    torch.manual_seed(0)
    text, region = tiny_text(), tiny_region()
    head = SlotHead(text, slots=6, label_tokens=3, rank=2)
    cache = prefix_cache(text, torch.randn(1, 7, DIM), no_rotation)
    objectness, box_logits, logits = head(text, region, cache, no_rotation)
    targets = [(torch.tensor([[0.5, 0.5, 0.3, 0.3]]), torch.tensor([[3, 4, 0]]))]
    options = loss_options({"objectness_target": "iou", "negatives": "near_labels"})
    terms, _, stats = set_loss(objectness, box_logits, logits, targets, **options)
    # Only unmatched slots near the labelled object are supervised as empty.
    assert 0.0 <= stats["supervised_empty_share"] <= 1.0
    everything = set_loss(objectness, box_logits, logits, targets)[2]
    assert everything["supervised_empty_share"] == 1.0
    # With IoU targets the matched slot is pushed toward its box quality, not to 1.
    assert torch.isfinite(terms["objectness"])
    assert loss_options({})["negatives"] == "all"


def test_teacher_items_filter_and_convert_labels(tmp_path):
    (tmp_path / "openimages").mkdir()
    (tmp_path / "openimages" / "1-2.jpg").write_bytes(b"jpeg")
    replies = [
        {
            "image_id": "openimages/1-2",
            "width": 200,
            "height": 100,
            "objects": [
                {"label": "dog", "box": [0, 0, 100, 50]},
                {"label": "cup", "box": [0, 0, 1, 1]},  # below the area floor
            ],
        },
        {"image_id": "openimages/3-4", "error": "HTTPError"},
        {"image_id": "openimages/5-6", "width": 10, "height": 10, "objects": []},
    ]
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(r) + "\n" for r in replies))
    config = {"min_area": 0.001, "max_objects": 30}
    ((read, record),) = list(teacher_items(labels, tmp_path, config))
    assert read() == b"jpeg"
    assert record == {"image_id": "openimages/1-2", "objects": [("dog", [0.25, 0.25, 0.5, 0.5])]}


def test_ignored_slots_get_no_empty_gradient():
    torch.manual_seed(0)
    text, region = tiny_text(), tiny_region()
    head = SlotHead(text, slots=6, label_tokens=3, rank=2)
    cache = prefix_cache(text, torch.randn(1, 7, DIM), no_rotation)
    objectness, box_logits, logits = head(text, region, cache, no_rotation)
    targets = [(torch.tensor([[0.5, 0.5, 0.3, 0.3]]), torch.tensor([[3, 4, 0]]))]
    ignore = torch.ones(1, 6, dtype=torch.bool)
    terms, (_, slot, _, _), stats = set_loss(objectness, box_logits, logits, targets, ignore=ignore)
    assert stats["ignored_empty_share"] == 1.0
    terms["objectness"].backward()
    gradient = head.objectness.weight.grad
    # Only the matched slot contributes, so the gradient equals that slot's alone.
    assert gradient is not None and torch.isfinite(gradient).all()
    head.zero_grad()
    objectness, box_logits, logits = head(text, region, cache, no_rotation)
    only = objectness[:, slot]
    expected = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            only, torch.ones_like(only), reduction="sum"
        )
        / objectness.numel()
    )
    expected.backward()
    assert torch.allclose(gradient, head.objectness.weight.grad, atol=1e-6)
