import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from distillation.export import convert

ROOT = Path(__file__).resolve().parents[2]
for name, folder in [("model_10m", "10m"), ("model_100m", "100m")]:
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "training" / folder / "__init__.py",
        submodule_search_locations=[str(ROOT / "training" / folder)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
from model_100m.model import GPT, ModelConfig


class ExportTests(unittest.TestCase):
    def test_full_logits_and_cached_decode(self):
        torch.manual_seed(42)
        native = GPT(
            ModelConfig(vocab_size=32, n_layer=2, n_head=2, n_embd=16, block_size=32)
        ).eval()
        exported, report = convert(native)
        self.assertLess(report["max_abs_logit_error"], 1e-5)
        tokens = torch.randint(0, 32, (2, 13))
        with torch.no_grad():
            expected = native(tokens)
            actual = exported(tokens, use_cache=True)
            torch.testing.assert_close(expected, actual.logits, rtol=1e-5, atol=1e-6)
            suffix = torch.randint(0, 32, (2, 3))
            cached = exported(
                suffix, past_key_values=actual.past_key_values, use_cache=True
            ).logits
            torch.testing.assert_close(
                native(torch.cat([tokens, suffix], 1))[:, -3:],
                cached,
                rtol=1e-5,
                atol=1e-6,
            )
        self.assertEqual(
            exported.model.embed_tokens.weight.data_ptr(),
            exported.lm_head.weight.data_ptr(),
        )
