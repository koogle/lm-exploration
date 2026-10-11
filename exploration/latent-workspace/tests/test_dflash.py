from dataclasses import replace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from latent_workspace.budget import reserve
from latent_workspace.dflash import (
    DFlashDrafter,
    DraftConfig,
    block_loss,
    features,
    pack_blocks,
    verified_generate,
)
from latent_workspace.dflash_data import PromptRegistry
from latent_workspace.dflash_train import greedy_generate


def fixture():
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
    return target, DFlashDrafter(config, cfg)


def test_packed_attention_excludes_anchor_future_context_and_other_blocks():
    tokens = torch.arange(12)[None]
    noisy, labels, positions, mask, selected = pack_blocks(tokens, torch.tensor([[3, 7]]), 4, 30)
    assert noisy.tolist() == [[3, 30, 30, 30, 7, 30, 30, 30]]
    assert positions.tolist() == [[3, 4, 5, 6, 7, 8, 9, 10]]
    assert labels.tolist() == positions.tolist()
    assert selected.tolist() == [False, True, True, True, False, True, True, True]
    assert mask[0, 0, :4, :3].all()
    assert not mask[0, 0, :4, 3:12].any()
    assert mask[0, 0, :4, 12:16].all()
    assert not mask[0, 0, :4, 16:].any()


def test_future_features_and_other_anchor_cannot_leak_into_earlier_block():
    target, draft = fixture()
    draft.eval()
    tokens = torch.randint(31, (1, 12))
    memory = features(target.model(tokens, output_hidden_states=True), draft.cfg.layers)
    anchors = torch.tensor([[3, 7]])
    a = block_loss(draft, target, tokens, memory, anchors, 30)
    altered = memory.clone()
    altered[:, 3:] = torch.randn_like(altered[:, 3:]) * 20
    changed = tokens.clone()
    changed[:, 7:] = (changed[:, 7:] + 1) % 31
    b = block_loss(draft, target, changed, altered, anchors, 30)
    torch.testing.assert_close(a[1][:, :3], b[1][:, :3], atol=0, rtol=0)
    a[0].backward()
    assert all(p.grad is None for p in target.parameters())
    assert draft.fc.weight.grad.abs().sum() > 0


def test_incremental_draft_context_cache_matches_full_cache():
    target, draft = fixture()
    draft.eval()
    tokens = torch.randint(31, (1, 10))
    memory = features(target.model(tokens, output_hidden_states=True), draft.cfg.layers)
    positions = torch.arange(10)[None]
    full = draft.context_cache(memory, positions)
    first = draft.context_cache(memory[:, :6], positions[:, :6])
    incremental = draft.context_cache(memory[:, 6:], positions[:, 6:], first)
    for a, b in zip(full, incremental):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 9, 17])
def test_verified_generation_equals_cached_target_with_rejections_and_length_caps(length):
    target, draft = fixture()
    prompt = torch.tensor([[4, 7, 9, 1]])
    expected = greedy_generate(target, prompt, length)
    actual = verified_generate(draft, target, prompt, length, 30)
    assert actual["tokens"] == expected["tokens"]
    assert len(actual["tokens"]) == length
    assert all(0 <= n <= 3 for n in actual["accepted_new_draft_tokens"])
    assert any(n < 3 for n in actual["accepted_new_draft_tokens"]) or length == 1


def test_verified_generation_stops_on_target_eos():
    target, draft = fixture()
    prompt = torch.tensor([[4, 7, 9, 1]])
    token = greedy_generate(target, prompt, 1)["tokens"][0]
    result = verified_generate(draft, target, prompt, 8, 30, token)
    assert result["tokens"] == [token]
    assert result["accepted_new_draft_tokens"] == []


def test_prompt_registry_quarantines_cross_split_near_duplicates():
    registry = PromptRegistry()
    first = {"prompt_ids": list(range(128)), "group": "a", "source_hash": "a", "split": "train"}
    assert registry.add(first)
    second = {
        **first,
        "group": "b",
        "source_hash": "b",
        "split": "validation",
        "prompt_ids": list(range(127)) + [999],
    }
    assert not registry.add(second)
    assert registry.rejected["cross_split_near_duplicate_prompt"] == 1


def test_short_worker_reservation_matches_actual_timeout_and_keeps_cap(tmp_path):
    path = tmp_path / "budget.json"
    assert reserve(path, "short", "L40S", timeout_seconds=900) == 1.25
    with pytest.raises(ValueError, match="timeout"):
        reserve(path, "bad", "L40S", timeout_seconds=0)
    with pytest.raises(ValueError, match="timeout"):
        reserve(path, "too-long", "L40S", timeout_seconds=3601)


def test_verifier_commits_fully_accepted_blocks_without_skipping_bonus_tokens():
    target, draft = fixture()
    prompt = torch.tensor([[4, 7, 9, 1]])
    expected = greedy_generate(target, prompt, 12)["tokens"]
    full = torch.cat((prompt, torch.tensor([expected])), dim=1)
    with torch.no_grad():
        oracle_hidden = target.model(full).last_hidden_state
    # Oracle is only a verifier fixture: it deliberately knows the correct future.
    draft.forward = lambda embeddings, positions, cache, mask=None: oracle_hidden[
        :, positions[0] - 1
    ]
    result = verified_generate(draft, target, prompt, 12, 30)
    assert result["tokens"] == expected
    assert result["accepted_new_draft_tokens"] == [3, 3, 3]
    assert result["verification_calls"] == 3


def test_continuation_restores_optimizer_and_exact_next_training_update():
    import copy

    from latent_workspace.dflash_train import restore_continuation

    target, draft = fixture()
    cfg = draft.cfg
    tokens = torch.randint(31, (1, 12))
    memory = features(target.model(tokens, output_hidden_states=True), cfg.layers)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=cfg.learning_rate)
    generator = torch.Generator().manual_seed(17)

    def update(head, opt, rng):
        anchors = torch.randint(3, 8, (1, 2), generator=rng)
        opt.zero_grad(set_to_none=True)
        block_loss(head, target, tokens, memory, anchors, 30)[0].backward()
        opt.step()

    update(draft, optimizer, generator)
    state = copy.deepcopy(
        {
            "step": 1,
            "drafter": draft.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "sampler_rng": generator.get_state(),
            "cuda_rng": [],
        }
    )
    update(draft, optimizer, generator)
    restored = DFlashDrafter(target.config, cfg)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0)
    restored_generator = torch.Generator()
    previous = {"config": cfg.to_dict(), "curve": [{"step": 1}]}
    restore_continuation(cfg, previous, state, restored, restored_optimizer, restored_generator)
    update(restored, restored_optimizer, restored_generator)
    for key, tensor in draft.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="block_size"):
        restore_continuation(
            replace(cfg, block_size=8),
            previous,
            state,
            restored,
            restored_optimizer,
            restored_generator,
        )


def test_partial_rotary_preserves_unrotated_channels():
    from latent_workspace.dflash import rotate

    x = torch.randn(2, 3, 5, 16, requires_grad=True)
    cos = torch.zeros(2, 5, 4)
    sin = torch.ones(2, 5, 4)
    y = rotate(x, cos, sin)
    torch.testing.assert_close(y[..., 4:], x[..., 4:])
    torch.testing.assert_close(y[..., :2], -x[..., 2:4])
    torch.testing.assert_close(y[..., 2:4], x[..., :2])
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_hybrid_target_requires_quality_only():
    from dataclasses import replace

    cfg = replace(DraftConfig(), model="Qwen/Qwen3.5-2B")
    with pytest.raises(ValueError, match="quality-only"):
        cfg.validate()
    replace(cfg, quality_only=True).validate()


def test_qwen35_hybrid_features_are_causal_and_drafter_trains():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(9)
    config = Qwen3_5TextConfig(
        vocab_size=31,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 2],
        },
    )
    target = Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
    cfg = replace(
        DraftConfig(), model="Qwen/Qwen3.5-2B", quality_only=True, layers=(1, 3), block_size=4
    )
    draft = DFlashDrafter(config, cfg)
    tokens = torch.randint(31, (1, 12))
    changed = tokens.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 31
    memory = features(target.model(tokens, use_cache=False, output_hidden_states=True), cfg.layers)
    other = features(target.model(changed, use_cache=False, output_hidden_states=True), cfg.layers)
    torch.testing.assert_close(memory[:, :6], other[:, :6])
    loss, _, _, _ = block_loss(draft, target, tokens, memory, torch.tensor([[6]]), 30)
    loss.backward()
    assert torch.isfinite(loss)
    assert draft.fc.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())


def test_qk_normalization_preserves_common_initialization_and_causal_cache():
    target, original = fixture()
    cfg = replace(original.cfg, qk_norm=True)
    torch.manual_seed(42)
    plain = DFlashDrafter(target.config, original.cfg)
    torch.manual_seed(42)
    normed = DFlashDrafter(target.config, cfg)
    for name, value in plain.state_dict().items():
        torch.testing.assert_close(value, normed.state_dict()[name], atol=0, rtol=0)
    tokens = torch.randint(31, (1, 12))
    memory = features(target.model(tokens, output_hidden_states=True), cfg.layers)
    anchors = torch.tensor([[4, 8]])
    a = block_loss(normed, target, tokens, memory, anchors, 30)
    altered = memory.clone()
    altered[:, 4:] = torch.randn_like(altered[:, 4:]) * 20
    b = block_loss(normed, target, tokens, altered, anchors, 30)
    torch.testing.assert_close(a[1][:, :3], b[1][:, :3], atol=0, rtol=0)
    a[0].backward()
    assert normed.layers[0].q_norm.weight.grad.abs().sum() > 0
    assert normed.layers[0].k_norm.weight.grad.abs().sum() > 0
    assert normed.fc.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())
    full = normed.context_cache(memory, torch.arange(12)[None])
    prefix = normed.context_cache(memory[:, :5], torch.arange(5)[None])
    incremental = normed.context_cache(memory[:, 5:], torch.arange(5, 12)[None], prefix)
    for layer_full, layer_incremental in zip(full, incremental):
        for x, y in zip(layer_full, layer_incremental):
            torch.testing.assert_close(x, y)
    prompt = tokens[:, :4]
    assert (
        verified_generate(normed, target, prompt, 12, 30)["tokens"]
        == greedy_generate(target, prompt, 12)["tokens"]
    )
