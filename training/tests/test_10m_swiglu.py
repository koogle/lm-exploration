"""Correctness checks only: no dataset download or training worker launch."""

import unittest

import torch
from model_10m import model as vanilla
from model_10m_swiglu import model as variant
from torch.nn import functional as F

from generate import load_model
from trainer.checkpoints import runtime_provenance
from trainer.recipe import load_recipe


class SwiGLUChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_outputs_and_gradients_against_explicit_gate(self):
        mlp = variant.SwiGLU(24).double()
        x = torch.randn(2, 3, 24, dtype=torch.float64, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        actual = mlp(x)
        gate = F.linear(reference_x, mlp.gate.weight, mlp.gate.bias)
        up = F.linear(reference_x, mlp.up.weight, mlp.up.bias)
        expected = F.linear(gate * torch.sigmoid(gate) * up, mlp.down.weight, mlp.down.bias)
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, (x, *mlp.parameters()), upstream)
        expected_grad = torch.autograd.grad(expected, (reference_x, *mlp.parameters()), upstream)
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b)

    def test_budget_biases_and_only_mlp_changes(self):
        mlp = variant.SwiGLU(384)
        self.assertEqual(mlp.hidden_width, 1024)
        self.assertEqual(sum(p.numel() for p in mlp.parameters() if p.ndim == 2), 8 * 384**2)
        self.assertTrue(all(layer.bias is not None for layer in (mlp.gate, mlp.up, mlp.down)))
        self.assertEqual(variant.SwiGLU(16).hidden_width, 43)
        config = variant.ModelConfig(7, 8, 2, 3, 24)
        torch.manual_seed(42)
        baseline = vanilla.GPT(config)
        torch.manual_seed(42)
        model = variant.GPT(config)
        for name, parameter in baseline.named_parameters():
            if ".mlp." not in name:
                torch.testing.assert_close(parameter, model.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(all(isinstance(block.mlp, variant.SwiGLU) for block in model.blocks))
        self.assertTrue(
            all(isinstance(block.mlp_norm, torch.nn.LayerNorm) for block in model.blocks)
        )
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
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10_798_848)
        control = load_recipe("recipes/shakespeare-10m.json")
        recipe = load_recipe("recipes/shakespeare-10m-swiglu.json")
        for key in ("training", "backend", "dataset", "model_overrides"):
            self.assertEqual(getattr(control, key), getattr(recipe, key))
        config = variant.ModelConfig(7, 8, 1, 3, 24)
        model = variant.GPT(config)
        checkpoint = {
            "config": {"model_name": "model_10m_swiglu", "model": vars(config)},
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
