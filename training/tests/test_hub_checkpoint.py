import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from model_100m.model import GPT, ModelConfig
from safetensors.torch import load_file
from training_data.subword import digest

from trainer.hub_checkpoint import export_checkpoint


class HubCheckpointTests(unittest.TestCase):
    def test_strict_export_preserves_weights_without_training_state(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "run"
            training = Path(__file__).resolve().parents[1]
            for folder, package in (("100m", "model_100m"), ("10m", "model_10m")):
                shutil.copytree(training / folder, source / "source" / package)
            (source / "tokenizer").mkdir()
            (source / "tokenizer/tokenizer.json").write_text("{}")
            (source / "data-manifest.json").write_text("{}")
            model = GPT(ModelConfig(vocab_size=32768, block_size=16, n_layer=1, n_head=2, n_embd=8))
            saved = {
                "format": "scholarly-ddp-v1",
                "config": {"n_layer": 1, "n_head": 2, "n_embd": 8, "context": 16},
                "model": model.state_dict(),
                "optimizer": {"sensitive": "optimizer"},
                "rng": {"sensitive": "rng"},
                "data": {"loss_token_exposures": {"one": 123}},
                "step": 2,
                "tokens": 123,
                "parameters": sum(p.numel() for p in model.parameters()),
                "source_sha256": {"model": digest(source / "source/model_100m/model.py")},
            }
            checkpoint = source / "step-000000002.pt"
            torch.save(saved, checkpoint)
            result = export_checkpoint(checkpoint, root / "export", source, "test")
            self.assertEqual(result["tokens"], 123)
            exported = load_file(str(root / "export/model.safetensors"))
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, exported[name]))
            model.load_state_dict(exported, strict=True)
            self.assertEqual(
                model.lm_head.weight.data_ptr(), model.token_embedding.weight.data_ptr()
            )
            provenance = json.loads((root / "export/provenance.json").read_text())
            for name in ("optimizer", "rng", "model", "data"):
                self.assertNotIn(name, provenance)
            self.assertEqual(
                json.loads((root / "export/config.json").read_text())["block_size"], 16
            )
            saved["data"]["loss_token_exposures"]["one"] = 124
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, "Exposure counters"):
                export_checkpoint(checkpoint, root / "invalid", source, "test")
