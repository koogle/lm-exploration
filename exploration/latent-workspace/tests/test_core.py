from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latent_workspace.budget import CAP_USD, reserve
from latent_workspace.config import Config
from latent_workspace.data import documents, make_windows
from latent_workspace.models import BlockHead, corrupt, loss_and_logits, sample_block


def tiny_config():
    return Config(
        layers=(1, 2),
        context_length=24,
        memory_tokens=3,
        width=16,
        heads=2,
        decoder_layers=1,
        slots=2,
        refinement_rounds=1,
        variants=("parallel:final",),
    )


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(31, 12).requires_grad_(False)

    def embed(self, ids):
        return self.embedding(ids)

    def project(self, features):
        return features @ self.embedding.weight.T


@pytest.mark.parametrize(
    "variant",
    [
        "probe:1",
        "probe:attention1",
        "parallel:final",
        "diffusion:final",
        "diffusion:layer1",
        "diffusion:multi",
        "diffusion:attention1",
        "diffusion:attention_multi",
        "diffusion:jspace",
    ],
)
def test_gradients_stop_at_frozen_backbone(variant):
    cfg = tiny_config()
    head = BlockHead(12, cfg, variant)
    backbone = ToyBackbone()
    memory = torch.randn(2, 6, 3, 12)
    target = torch.randint(31, (2, 4))
    loss, logits, mask = loss_and_logits(head, backbone, memory, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.shape == (2, 4, 31)
    assert mask.any(-1).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert all(p.grad is None for p in backbone.parameters())
    generated = sample_block(head, backbone, memory)
    assert generated.shape == target.shape
    assert (generated >= 0).all() and (generated < 31).all()


def test_all_masked_logits_cannot_see_targets():
    cfg = tiny_config()
    head = BlockHead(12, cfg, "diffusion:multi").eval()
    backbone = ToyBackbone()
    memory = torch.randn(2, 6, 3, 12)
    a = loss_and_logits(
        head, backbone, memory, torch.zeros(2, 4, dtype=torch.long), all_masked=True
    )[1]
    b = loss_and_logits(
        head, backbone, memory, torch.ones(2, 4, dtype=torch.long), all_masked=True
    )[1]
    torch.testing.assert_close(a, b)


def test_selected_conditioning_ignores_other_channels():
    cfg = tiny_config()
    head = BlockHead(12, cfg, "parallel:layer1").eval()
    memory = torch.randn(2, 6, 3, 12)
    changed = memory.clone()
    changed[:, 1:] += 100
    torch.testing.assert_close(head(memory), head(changed))


def test_masks_include_full_and_partial_without_empty():
    mask, rate = corrupt(torch.zeros(500, 4, dtype=torch.long))
    assert mask.any(-1).all()
    assert set(mask.sum(-1).tolist()) == {1, 2, 3, 4}
    torch.testing.assert_close(rate, mask.float().mean(-1))


def test_document_split_precedes_windowing():
    rows = [
        {"text": t}
        for t in [
            " = Article one = ",
            "first paragraph",
            " = = Section = = ",
            "second paragraph",
            " = Article two = ",
            "third paragraph",
        ]
    ]
    texts = list(documents(rows))
    assert len(texts) == 2
    assert "second paragraph" in texts[0]
    tokenizer = SimpleNamespace(encode=lambda t, **kw: [ord(c) for c in t])
    held, hashes = make_windows([texts[0]], tokenizer, 4, 2, 100)
    train, other = make_windows(texts, tokenizer, 4, 2, 100, hashes)
    assert not hashes & other
    assert held.shape[1] == train.shape[1] == 6


def test_budget_fail_closed_and_durable(tmp_path):
    path = tmp_path / "budget.json"
    reserve(path, "first", "H100")
    with pytest.raises(ValueError, match="already reserved"):
        reserve(path, "first", "H100")
    for i in range(int(CAP_USD // 12) - 1):
        reserve(path, str(i), "H100")
    with pytest.raises(ValueError, match="cap reached"):
        reserve(path, "over", "H100")


def test_invalid_configs():
    with pytest.raises(ValueError):
        replace(tiny_config(), block_size=0).validate()
    with pytest.raises(ValueError):
        replace(tiny_config(), variants=("diffusion:layer99",)).validate()


@pytest.mark.parametrize("length", [4, 8, 16])
@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_sampling_commits_every_position_without_overwriting(length, steps):
    class FixedHead:
        kind = "diffusion"
        cfg = SimpleNamespace(block_size=length, diffusion_steps=steps)

        def __init__(self):
            self.previous_mask = None

        def __call__(self, memory, embeddings, mask, noise):
            if self.previous_mask is not None:
                assert not (mask & ~self.previous_mask).any()
            self.previous_mask = mask.clone()
            expected = torch.arange(1, length + 1)[None].expand(2, -1)
            return torch.nn.functional.one_hot(expected, 31).float() * 10

    backbone = SimpleNamespace(embed=lambda ids: ids, project=lambda x: x)
    result = sample_block(FixedHead(), backbone, torch.zeros(2, 1, 1, 1))
    torch.testing.assert_close(result, torch.arange(1, length + 1)[None].expand(2, -1))
