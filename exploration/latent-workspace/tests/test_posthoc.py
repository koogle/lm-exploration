import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_backbone import tiny_backbone
from test_core import tiny_config

from latent_workspace import posthoc
from latent_workspace.models import BlockHead


@pytest.mark.parametrize("evaluate_test", (False, True))
def test_saved_head_evaluation_opens_only_requested_split(tmp_path, monkeypatch, evaluate_test):
    cfg = replace(tiny_config(), variants=("diffusion:multi",))
    backbone = tiny_backbone()
    backbone.cfg = cfg
    backbone.tokenizer = SimpleNamespace(decode=lambda ids: "")
    source = tmp_path / "source"
    source.mkdir()
    (source / "study.json").write_text(
        json.dumps({"completed": True, "config": cfg.to_dict(), "curves": {"12": {}}})
    )
    rows = {
        "validation": torch.zeros(3, 28, dtype=torch.long),
        "test": torch.ones(3, 28, dtype=torch.long),
    }
    ranges = {
        split: {d: (i, i + 1) for i, d in enumerate(("prose", "code", "translation"))}
        for split in rows
    }
    torch.save({"tensors": rows, "metadata": {"domain_ranges": ranges}}, source / "corpus.pt")
    head = BlockHead(backbone.hidden_size, cfg, "diffusion:multi")
    torch.save(
        {"head": head.state_dict(), "step": 2, "variant": head.variant}, source / "size-12-best.pt"
    )
    seen = []

    def evaluate(head, backbone, data, *args, **kwargs):
        seen.append(data.clone())
        return {}

    monkeypatch.setattr(posthoc, "FrozenBackbone", lambda *args: backbone)
    monkeypatch.setattr(posthoc, "evaluate", evaluate)
    monkeypatch.setattr(posthoc, "generate", lambda *args, **kwargs: [])
    request = replace(cfg, study_parent=str(source), evaluate_test=evaluate_test)
    if evaluate_test:
        with pytest.raises(ValueError, match="frozen source report hash"):
            posthoc.evaluate_study(request, tmp_path / "unfrozen")
        request.study_eval_hashes = {
            str(source): hashlib.sha256((source / "study.json").read_bytes()).hexdigest()
        }
    result = posthoc.evaluate_study(request, tmp_path / "evaluation")
    assert len(seen) == 3
    assert all(torch.all(x == int(evaluate_test)) for x in seen)
    assert result["evaluation_split"] == ("test" if evaluate_test else "validation")
    assert result["heads"]["12"]["step"] == 2
