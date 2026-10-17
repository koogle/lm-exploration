"""Correctness checks only: no dataset download or training worker launch."""

import unittest

import torch
from model_10m import model as vanilla
from model_10m_rmsnorm import model as variant
from torch.nn import functional as F

from generate import load_model
from trainer.checkpoints import runtime_provenance
from trainer.recipe import load_recipe


class RMSNormChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_outputs_and_gradients_against_pytorch(self):
        norm = variant.RMSNorm(8).double()
        norm.weight.data.copy_(torch.linspace(0.5, 1.5, 8))
        x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        reference_weight = norm.weight.detach().clone().requires_grad_()
        actual = norm(x)
        expected = F.rms_norm(reference_x, (8,), reference_weight, eps=1e-5)
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(x.grad, reference_x.grad)
        torch.testing.assert_close(norm.weight.grad, reference_weight.grad)
        self.assertTrue(torch.autograd.gradcheck(norm, (x,)))

    def test_zero_and_low_precision_inputs(self):
        norm = variant.RMSNorm(8)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.zeros(2, 3, 8, dtype=dtype, requires_grad=True)
            out = norm(x)
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(torch.isfinite(out).all())
            out.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_only_normalization_changes(self):
        config = variant.ModelConfig(7, 8, 2, 3, 24)
        baseline = vanilla.GPT(config)
        torch.manual_seed(42)
        model = variant.GPT(config)
        for name, parameter in baseline.named_parameters():
            if "norm" not in name:
                torch.testing.assert_close(parameter, model.state_dict()[name], rtol=0, atol=0)
        norms = [module for module in model.modules() if isinstance(module, variant.RMSNorm)]
        self.assertEqual(len(norms), 2 * config.n_layer + 1)
        self.assertTrue(all(norm.eps == 1e-5 for norm in norms))
        self.assertFalse(any(isinstance(module, torch.nn.LayerNorm) for module in model.modules()))
        tokens = torch.randint(7, (2, 8))
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 7
        torch.testing.assert_close(model(tokens)[:, :4], model(changed)[:, :4], rtol=0, atol=0)
        F.cross_entropy(model(tokens).flatten(0, 1), tokens.flatten()).backward()
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        )

    def test_count_recipe_loading_and_provenance(self):
        with torch.device("meta"):
            baseline = vanilla.GPT(vanilla.ModelConfig(65))
            model = variant.GPT(variant.ModelConfig(65))
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), 10_795_776)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10_790_784)
        control = load_recipe("recipes/shakespeare-10m.json")
        recipe = load_recipe("recipes/shakespeare-10m-rmsnorm.json")
        for key in ("training", "backend", "dataset", "model_overrides"):
            self.assertEqual(getattr(control, key), getattr(recipe, key))
        config = variant.ModelConfig(7, 8, 1, 3, 24)
        model = variant.GPT(config)
        checkpoint = {
            "config": {"model_name": "model_10m_rmsnorm", "model": vars(config)},
            "model": model.state_dict(),
        }
        tokens = torch.randint(7, (1, 8))
        torch.testing.assert_close(model(tokens), load_model(checkpoint)(tokens), rtol=0, atol=0)
        sources = runtime_provenance("cpu", variant.__file__, variant.BASE_MODEL_SOURCE)[
            "source_hashes"
        ]
        self.assertIn("model", sources)
        self.assertIn("model_definition", sources)


if __name__ == "__main__":
    unittest.main()
