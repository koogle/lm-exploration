"""A checkpoint must reproduce the next update, not just load without errors."""

from types import SimpleNamespace

import torch
from test_core import ToyBackbone, tiny_config

from latent_workspace.models import BlockHead, loss_and_logits
from latent_workspace.train import checkpoint


def test_optimizer_and_rng_resume_reproduces_next_update(tmp_path):
    torch.manual_seed(12)
    cfg = tiny_config()
    backbone = ToyBackbone()
    head = BlockHead(12, cfg, "diffusion:multi")
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    sampler = torch.Generator().manual_seed(20)
    memory = torch.randn(5, 4, 3, 12)
    targets = torch.randint(31, (5, 4))

    def update(model, opt, rng):
        idx = torch.randint(5, (2,), generator=rng)
        opt.zero_grad()
        loss, _, _ = loss_and_logits(model, backbone, memory[idx], targets[idx])
        loss.backward()
        opt.step()
        return loss.detach()

    update(head, optimizer, sampler)
    path = tmp_path / "state.pt"
    checkpoint(
        path,
        head,
        optimizer,
        1,
        cfg,
        SimpleNamespace(revision="frozen-test", jacobians=None),
        sampler,
    )
    expected_loss = update(head, optimizer, sampler)
    restored = BlockHead(12, cfg, "diffusion:multi")
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.001)
    state = torch.load(path, weights_only=True)
    restored.load_state_dict(state["head"])
    restored_opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    sampler.set_state(state["sampler_rng"])
    actual_loss = update(restored, restored_opt, sampler)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    for expected, actual in zip(head.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_training_pipeline_and_completed_resume(tmp_path, monkeypatch):
    from dataclasses import replace

    from test_backbone import tiny_backbone

    from latent_workspace import train

    cfg = replace(
        tiny_config(),
        steps=2,
        batch_size=2,
        eval_every=1,
        variants=("diffusion:multi",),
        max_seconds=60,
    )
    ids = torch.randint(31, (8, 28))

    class Tokenizer:
        eos_token_id = 30

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 4]]))

        def decode(self, tokens):
            return " ".join(map(str, tokens))

    def backbone_factory(config, device):
        backbone = tiny_backbone()
        backbone.cfg = config
        backbone.revision = "toy-model-commit"
        backbone.tokenizer = Tokenizer()
        return backbone

    monkeypatch.setattr(train, "FrozenBackbone", backbone_factory)
    monkeypatch.setattr(
        train,
        "load_data",
        lambda cfg, tok: (ids[:6], ids[6:], {"dataset_revision": "toy-data-commit"}),
    )
    monkeypatch.setattr(train, "benchmark", lambda *a, **kw: {"test_only": True})
    first = train.run(cfg, tmp_path)
    resumed = train.run(cfg, tmp_path, resume=True)
    assert first["variants"]["diffusion:multi"]["completed"]
    assert (
        first["variants"]["diffusion:multi"]["held_out"]
        == resumed["variants"]["diffusion:multi"]["held_out"]
    )
    assert resumed["variants"]["diffusion:multi"]["steps"] == 2
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "results.json").exists()
