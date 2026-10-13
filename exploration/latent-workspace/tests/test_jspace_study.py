"""Check competent initialization and causal masking of the semantic side channel."""

import importlib
from dataclasses import replace
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from latent_workspace.dflash import DFlashDrafter, DraftConfig, block_loss, features


def fixture(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("jspace_study")
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=4,
        max_position_embeddings=128,
        eos_token_id=30,
    )
    target = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    cfg = replace(DraftConfig(), layers=(1, 3), block_size=4)
    original = DFlashDrafter(config, cfg).eval()
    augmented = module.AugmentedDrafter(config, cfg, original.state_dict()).eval()
    tokens = torch.randint(31, (1, 12))
    raw = features(target.model(tokens, output_hidden_states=True), cfg.layers)
    return target, original, augmented, tokens, raw


def test_zero_adapter_preserves_capable_predictions_and_has_gradients(monkeypatch):
    target, original, augmented, tokens, raw = fixture(monkeypatch)
    anchors = torch.tensor([[3, 7]])
    baseline = block_loss(original, target, tokens, raw, anchors, 30)
    side = torch.randn_like(raw)
    actual = block_loss(augmented, target, tokens, torch.cat((raw, side), -1), anchors, 30)
    torch.testing.assert_close(baseline[1], actual[1], atol=0, rtol=0)
    assert torch.equal(baseline[2], actual[2])
    actual[0].backward()
    assert augmented.correction.weight.grad.abs().sum() > 0
    assert augmented.fc.weight.grad is None
    assert all(p.grad is None for p in target.parameters())


def test_trained_adapter_cannot_access_anchor_or_future_features(monkeypatch):
    target, _, augmented, tokens, raw = fixture(monkeypatch)
    with torch.no_grad():
        augmented.correction.weight.normal_(std=0.03)
    memory = torch.cat((raw, torch.randn_like(raw)), -1)
    anchors = torch.tensor([[3, 7]])
    a = block_loss(augmented, target, tokens, memory, anchors, 30)
    altered = memory.clone()
    altered[:, 3:] = torch.randn_like(altered[:, 3:]) * 20
    b = block_loss(augmented, target, tokens, altered, anchors, 30)
    torch.testing.assert_close(a[1][:, :3], b[1][:, :3], atol=0, rtol=0)


def test_auxiliary_incremental_context_cache_matches_full(monkeypatch):
    _, _, augmented, _, raw = fixture(monkeypatch)
    with torch.no_grad():
        augmented.correction.weight.normal_(std=0.03)
    memory = torch.cat((raw, torch.randn_like(raw)), -1)
    positions = torch.arange(12)[None]
    full = augmented.context_cache(memory, positions)
    prefix = augmented.context_cache(memory[:, :7], positions[:, :7])
    incremental = augmented.context_cache(memory[:, 7:], positions[:, 7:], prefix)
    for first, second in zip(full, incremental):
        for x, y in zip(first, second):
            torch.testing.assert_close(x, y)


def test_paired_bootstrap_preserves_domain_weights_and_identical_pairs(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("jspace_study")
    metrics = {}
    for arm, offset in [("raw", 0), ("jspace", 1), ("random", 0.25)]:
        rows = [
            {
                "group": f"{domain}-{i}",
                "domain": domain,
                "blocks": i + 1,
                "accepted_beyond_anchor": i + offset,
            }
            for domain in ("prose", "code", "translation")
            for i in range(3)
        ]
        metrics[arm] = {"examples": rows, "accepted_beyond_anchor": 4 / 3 + offset}
    intervals = module.paired_intervals(metrics, iterations=200)
    assert abs(intervals["jspace-minus-raw"]["delta"] - 1) < 1e-12
    assert all(abs(x - 1) < 1e-12 for x in intervals["jspace-minus-raw"]["ci95"])
    metrics["jspace"]["examples"][0]["group"] = "wrong-source"
    import pytest

    with pytest.raises(ValueError, match="source identities"):
        module.paired_intervals(metrics, iterations=200)
