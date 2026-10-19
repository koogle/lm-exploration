import copy
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from training_data.subword import TOKENIZER_SHA256

from trainer.continued_pretraining import coding_double_mix, migrate_phase, token_targets
from trainer.distributed_config import DistributedConfig, validate_resume


class ContinuedPretrainingTests(unittest.TestCase):
    def test_exact_mix_and_integer_token_budget(self):
        original = {
            "code_python": {"weight_percent": 30},
            "web": {"weight_percent": 10},
            "math": {"weight_percent": 20},
            "papers": {"weight_percent": 15},
            "books": {"weight_percent": 25},
        }
        mix = coding_double_mix(original)
        self.assertEqual(mix["code_python"], 60)
        self.assertEqual(sum(mix.values()), 100)
        for name in ("web", "math", "papers", "books"):
            self.assertEqual(mix[name] / original[name]["weight_percent"], Fraction(4, 7))
        targets = token_targets(mix)
        self.assertEqual(sum(targets.values()), 5_000_000_000)
        self.assertEqual(targets["code_python"], 3_000_000_000)
        self.assertEqual(targets, token_targets(mix))

    def test_nested_phase_preserves_weights_adam_and_lifetime_count(self):
        torch.set_num_threads(1)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        config = DistributedConfig(
            world_size=8,
            global_windows=32,
            microbatch=2,
            context=8192,
            target_tokens=5_000_000_000,
            peak_lr=3e-5,
            warmup_tokens=1,
            min_lr_ratio=1.0,
        )
        parent = {
            "format": "scholarly-ddp-v1",
            "config": config.as_dict(),
            "data_identity": {"tokenizer_sha256": TOKENIZER_SHA256},
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "rng": list(range(8)),
            "tokens": 878610362,
            "step": 3838,
            "lineage": {"parent_tokens": 4999981176},
            "source_sha256": {"old": "hash"},
            "torch_version": str(torch.__version__),
        }
        identity = {"manifest_sha256": "new", "tokenizer_sha256": TOKENIZER_SHA256}
        data = SimpleNamespace(
            identity=identity, state_dict=lambda: {"fresh": True}, close=lambda: None
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.save(parent, root / "parent.pt")
            original = (root / "parent.pt").read_bytes()
            with (
                patch("model_100m.model.GPT", return_value=torch.nn.Linear(3, 2)),
                patch("training_data.raw_token_stream.RawTokenMixture", return_value=data),
            ):
                lineage = migrate_phase(
                    root / "parent.pt",
                    root / "seed.pt",
                    root / "manifest.json",
                    config,
                    {"target_tokens": 5_000_000_000},
                )
            seed = torch.load(root / "seed.pt", weights_only=False)
            self.assertEqual(lineage["parent_tokens"], 5878591538)
            self.assertEqual(lineage["immediate_parent_phase_tokens"], 878610362)
            self.assertEqual(seed["tokens"], 0)
            self.assertEqual(seed["step"], 0)
            self.assertEqual(seed["data"], {"fresh": True})
            self.assertEqual(seed["rng"], parent["rng"])
            self.assertEqual((root / "parent.pt").read_bytes(), original)
            for name, value in parent["model"].items():
                torch.testing.assert_close(seed["model"][name], value, rtol=0, atol=0)
            for key, state in parent["optimizer"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(
                        seed["optimizer"]["state"][key][name], value, rtol=0, atol=0
                    )
            validate_resume(seed, config, identity, seed["source_sha256"], str(torch.__version__))


if __name__ == "__main__":
    unittest.main()
