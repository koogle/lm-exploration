from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from latent_workspace.backbone import FrozenBackbone
from latent_workspace.config import Config
from latent_workspace.jspace import fit_jacobians, readout


def tiny_backbone():
    cfg = Config(
        layers=(1, 2),
        context_length=24,
        memory_tokens=3,
        jspace_prompts=1,
        jspace_dim_batch=4,
        jspace_top_k=5,
    )
    backbone = FrozenBackbone.__new__(FrozenBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.cfg = cfg
    backbone.model = (
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=31,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=128,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    backbone.hidden_size = 8
    backbone.jacobians = None
    backbone.tokenizer = SimpleNamespace(eos_token_id=30)
    return backbone


def test_actual_qwen_hooks_and_kv_cache_agree():
    backbone = tiny_backbone()
    prefix = torch.randint(31, (1, 24))
    next_block = torch.randint(31, (1, 4))
    memory, cache = backbone.encode(prefix)
    incremental, _ = backbone.encode(next_block, past_key_values=cache)
    full, _ = backbone.encode(torch.cat([prefix, next_block], -1))
    assert memory.shape == (1, 4, 3, 8)
    torch.testing.assert_close(full, incremental, atol=1e-5, rtol=1e-5)
    assert not memory.requires_grad
    assert len(backbone.model.model.layers[0]._forward_hooks) == 0


def test_exact_jacobian_matches_direct_autograd():
    import time

    backbone = tiny_backbone()
    prefix = torch.randint(31, (1, 24))
    fitted = fit_jacobians(backbone, prefix, deadline=time.monotonic() + 60)
    values = {}

    def first(module, args, output):
        h = (output[0] if isinstance(output, tuple) else output).detach().requires_grad_(True)
        values[1] = h
        return (h, *output[1:]) if isinstance(output, tuple) else h

    def last(module, args, output):
        values[2] = output[0] if isinstance(output, tuple) else output

    handles = [
        backbone.model.model.layers[0].register_forward_hook(first),
        backbone.model.model.layers[1].register_forward_hook(last),
    ]
    try:
        backbone.model.model(prefix, use_cache=False)
        expected = []
        for d in range(8):
            grad = torch.autograd.grad(values[2][:, 16:-1, d].sum(), values[1], retain_graph=True)[
                0
            ]
            expected.append(grad[0, 16:-1].mean(0))
    finally:
        for h in handles:
            h.remove()
    torch.testing.assert_close(fitted[0], torch.stack(expected), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fitted[1], torch.eye(8))
    backbone.jacobians = fitted
    memory, _ = backbone.encode(prefix)
    assert memory.shape == (1, 6, 3, 8)
    assert torch.isfinite(readout(backbone, memory[:, :2])).all()
    assert all(p.grad is None for p in backbone.parameters())


def test_frozen_readout_matches_teacher_and_padding_cache():
    backbone = tiny_backbone()
    prefix = torch.randint(31, (1, 24))
    memory, _ = backbone.encode(prefix)
    logits = backbone.model(prefix).logits[:, -1]
    torch.testing.assert_close(backbone.project(memory[:, 1, -1]), logits)
    padded = torch.cat([torch.full((1, 5), -1), prefix], -1)
    padded_memory, cache = backbone.encode(padded)
    torch.testing.assert_close(padded_memory, memory, atol=1e-5, rtol=1e-5)
    block = torch.randint(31, (1, 4))
    incremental, _ = backbone.encode(block, past_key_values=cache)
    full, _ = backbone.encode(torch.cat([prefix, block], -1))
    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("position_readout", [False, True])
@pytest.mark.parametrize("normalize_token_inputs", [False, True])
def test_residual_drafter_initializes_to_pretrained_logits_and_learns(
    position_readout, normalize_token_inputs
):
    from dataclasses import replace

    from test_core import tiny_config

    from latent_workspace.models import BlockHead, loss_and_logits

    backbone = tiny_backbone()
    cfg = replace(
        tiny_config(),
        residual_readout=True,
        position_readout=position_readout,
        normalize_token_inputs=normalize_token_inputs,
    )
    backbone.cfg = cfg
    prefix = torch.randint(31, (2, 24))
    target = torch.randint(31, (2, 4))
    memory, _ = backbone.encode(prefix)
    head = BlockHead(8, cfg, "diffusion:multi")
    loss, logits, _ = loss_and_logits(head, backbone, memory, target, all_masked=True)
    expected = backbone.model(prefix).logits[:, -1:, :].expand_as(logits)
    torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)
    loss.backward()
    projections = head.out if position_readout else [head.out]
    assert all(projection.weight.grad.abs().sum() > 0 for projection in projections)
    if position_readout:
        assert not torch.equal(head.out[0].weight.grad, head.out[1].weight.grad)
    assert all(p.grad is None for p in backbone.parameters())


def test_known_first_token_drafting_preserves_qwen_choice():
    from dataclasses import replace

    from test_core import tiny_config

    from latent_workspace.models import BlockHead, sample_block

    backbone = tiny_backbone()
    cfg = replace(tiny_config(), residual_readout=True, position_readout=True)
    backbone.cfg = cfg
    prefix = torch.randint(31, (2, 24))
    memory, _ = backbone.encode(prefix)
    head = BlockHead(8, cfg, "diffusion:multi")
    expected = backbone.model(prefix).logits[:, -1].argmax(-1)
    for rounds in (1, 2, 4):
        for ordered in (False, True):
            proposed = sample_block(
                head, backbone, memory, steps=rounds, anchor_first=True, left_to_right=ordered
            )
            torch.testing.assert_close(proposed[:, 0], expected)
            assert proposed.shape == (2, 4)


def test_cross_backbone_token_ids_and_segmentation_must_match():
    import json

    from latent_workspace.backbone import compatible_tokenizers

    def tokenizer(vocab, pre="same"):
        return SimpleNamespace(
            get_vocab=lambda: vocab,
            backend_tokenizer=SimpleNamespace(to_str=lambda: json.dumps({"pre_tokenizer": pre})),
        )

    source = tokenizer({"a": 0, "b": 1})
    target = tokenizer({"a": 0, "b": 1, "<extra>": 2})
    assert compatible_tokenizers(source, target)["all_source_token_ids_preserved"]
    with pytest.raises(ValueError, match="token IDs"):
        compatible_tokenizers(source, tokenizer({"a": 1, "b": 0}))
    with pytest.raises(ValueError, match="segmentation"):
        compatible_tokenizers(source, tokenizer({"a": 0, "b": 1}, "different"))
