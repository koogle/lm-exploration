import torch
from torch import nn

from parallel_detection.slots import (
    BINS,
    SlotHead,
    bins_of,
    boxes_of,
    diffusion_loss,
    prefix_cache,
    sample,
    set_loss,
    total,
)
from parallel_detection.train_slots import decode_names, kept_objects

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
    _, boxes, tokens = sample(head, text, region, cache, no_rotation, steps=1)
    assert boxes.shape == (1, 4, 4) and tokens.shape == (1, 4, 3)
    # A position revealed after the first pass keeps its value in every later pass.
    _, first_boxes, first_tokens = sample(head, text, region, cache, no_rotation, steps=1)
    _, later_boxes, later_tokens = sample(head, text, region, cache, no_rotation, steps=6)
    same = (first_tokens == later_tokens).sum() + (first_boxes == later_boxes).sum()
    assert same >= 4  # at least the first revealed position of each slot


def test_label_filter_and_decoding():
    objects = {"category": [0, 1], "bbox": [[0, 0, 50, 50], [0, 0, 2, 2]]}
    kept = kept_objects(objects, 100, 100, min_area=0.002, names=["person", "bicycle"])
    assert kept == [("person", [0.25, 0.25, 0.5, 0.5])]
    vocabulary = {"person": [5, 0, 0], "fire hydrant": [6, 7, 8]}
    tokens = torch.tensor([[[5, 0, 0], [6, 7, 0], [6, 7, 8]]])
    assert decode_names(tokens, vocabulary) == ["person", None, "fire hydrant"]
