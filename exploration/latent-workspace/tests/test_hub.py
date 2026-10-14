import json

import pytest
import torch
from test_dflash import fixture

from latent_workspace.hub import download_checkpoint, export_checkpoint, sha256


def test_export_preserves_draft_weights_without_training_data(tmp_path):
    target, draft = fixture()
    checkpoint = tmp_path / "training.pt"
    torch.save(
        {
            "drafter": draft.state_dict(),
            "config": draft.cfg.to_dict(),
            "step": 7,
            "mask_token_id": 30,
            "optimizer": {"private": "do not publish"},
            "examples": ["private training text"],
        },
        checkpoint,
    )
    output = export_checkpoint(checkpoint, tmp_path / "export", sha256(checkpoint))
    state = torch.load(output / "drafter.pt", weights_only=True)
    assert set(state) == {"drafter", "config", "step", "mask_token_id"}
    restored = type(draft)(target.config, draft.cfg)
    restored.load_state_dict(state["drafter"])
    for key, value in draft.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["inference_checkpoint_sha256"] == sha256(output / "drafter.pt")
    assert "from .dflash import" not in (output / "dflash_cli.py").read_text()


def test_export_checks_checksum_before_creating_folder(tmp_path):
    checkpoint = tmp_path / "invalid.pt"
    checkpoint.write_bytes(b"wrong file")
    with pytest.raises(ValueError, match="checksum"):
        export_checkpoint(checkpoint, tmp_path / "export", "0" * 64)
    assert not (tmp_path / "export").exists()


def test_hub_download_requires_revision():
    with pytest.raises(ValueError, match="revision"):
        download_checkpoint("account/model", None)
