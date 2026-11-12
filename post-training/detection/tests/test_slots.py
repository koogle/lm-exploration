import torch
from torch import nn

from parallel_detection.slots import (
    SlotHead,
    diffusion_loss,
    prefix_cache,
    sample_labels,
    set_loss,
    total,
)
from parallel_detection.train_slots import decode_names, kept_objects

DIM, VOCABULARY = 64, 50


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


def test_slot_head_trains_on_a_set_loss_and_leaves_the_backbone_frozen():
    torch.manual_seed(0)
    text = tiny_text()
    head = SlotHead(text, slots=5, label_tokens=3, rank=2)
    cache = prefix_cache(text, torch.randn(2, 7, DIM), no_rotation)
    objectness, boxes, logits = head(text, cache, no_rotation)
    assert objectness.shape == (2, 5) and boxes.shape == (2, 5, 4) and logits.shape == (2, 5, 3, 50)
    targets = [
        (
            torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]),
            torch.tensor([[3, 4, 0], [7, 0, 0]]),
        ),
        (torch.tensor([[0.7, 0.3, 0.4, 0.4]]), torch.tensor([[9, 9, 9]])),
    ]
    terms, matching, stats = set_loss(objectness, boxes, logits, targets)
    assert matching[0].tolist() == [0, 0, 1] and len(set(matching[1][:2].tolist())) == 2
    assert 0 <= stats["matched_iou"] <= 1
    generator = torch.Generator().manual_seed(0)
    loss = total(terms) + diffusion_loss(head, text, cache, no_rotation, matching, generator)
    loss.backward()
    assert head.queries.grad.abs().sum() > 0 and head.box[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in text.parameters())


def test_sampler_keeps_revealed_tokens_between_passes():
    torch.manual_seed(0)
    text = tiny_text()
    head = SlotHead(text, slots=4, label_tokens=3, rank=2)
    cache = prefix_cache(text, torch.randn(1, 7, DIM), no_rotation)
    _, _, one, _ = sample_labels(head, text, cache, no_rotation, steps=1)
    _, _, three, confidence = sample_labels(head, text, cache, no_rotation, steps=3)
    assert one.shape == three.shape == (1, 4, 3)
    # After three passes over three positions, two positions per slot were revealed.
    assert ((confidence == 1).sum(-1) == 2).all()


def test_label_filter_and_decoding():
    objects = {"category": [0, 1], "bbox": [[0, 0, 50, 50], [0, 0, 2, 2]]}
    kept = kept_objects(objects, 100, 100, min_area=0.002, names=["person", "bicycle"])
    assert kept == [("person", [0.25, 0.25, 0.5, 0.5])]
    vocabulary = {"person": [5, 0, 0], "fire hydrant": [6, 7, 8]}
    tokens = torch.tensor([[[5, 0, 0], [6, 7, 0], [6, 7, 8]]])
    assert decode_names(tokens, vocabulary) == ["person", None, "fire hydrant"]
