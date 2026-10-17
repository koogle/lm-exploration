"""Correctness checks only: no dataset download or training worker launch."""

import math
import unittest

import torch
from model_10m import model as vanilla
from model_10m_rope import model as variant
from torch.nn import functional as F

from generate import load_model
from trainer.checkpoints import runtime_provenance
from trainer.recipe import load_recipe


def complex_rotation(x, offset=0):
    pairs = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
    frequencies = 10000.0 ** (-torch.arange(0, x.size(-1), 2).double() / x.size(-1))
    angles = torch.outer(torch.arange(offset, offset + x.size(-2)).double(), frequencies)
    phase = torch.polar(torch.ones_like(angles), angles)
    return torch.view_as_real(pairs * phase).flatten(-2)


class RoPEChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_rotations_gradients_and_relative_positions(self):
        x = torch.randn(1, 2, 5, 8, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(variant.rotary(x, 7), complex_rotation(x, 7))
        torch.testing.assert_close(variant.rotary(x)[..., 0, :], x[..., 0, :])
        torch.testing.assert_close(variant.rotary(x).square().sum(-1), x.square().sum(-1))
        self.assertTrue(torch.autograd.gradcheck(variant.rotary, (x,)))
        key = torch.randn_like(x)
        scores = variant.rotary(x) @ variant.rotary(key).transpose(-2, -1)
        shifted = variant.rotary(x, 17) @ variant.rotary(key, 17).transpose(-2, -1)
        torch.testing.assert_close(scores, shifted)
        for dtype in (torch.float16, torch.bfloat16):
            self.assertEqual(variant.rotary(x.to(dtype)).dtype, dtype)

    def test_attention_outputs_and_gradients_against_reference(self):
        config = variant.ModelConfig(7, 8, 1, 3, 24)
        attention = variant.Attention(config).double()
        x = torch.randn(2, 5, 24, dtype=torch.float64, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        actual = attention(x)
        query, key, value = attention.qkv(reference_x).chunk(3, -1)
        query, key, value = [t.view(2, 5, 3, 8).transpose(1, 2) for t in (query, key, value)]
        scores = complex_rotation(query) @ complex_rotation(key).transpose(-2, -1) / math.sqrt(8)
        mask = torch.ones(5, 5, dtype=torch.bool).tril()
        attended = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ value
        expected = attention.projection(attended.transpose(1, 2).reshape(2, 5, 24))
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, (x, *attention.parameters()), upstream)
        expected_grad = torch.autograd.grad(
            expected, (reference_x, *attention.parameters()), upstream
        )
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b)

    def test_causality_bounds_and_vanilla_components(self):
        config = variant.ModelConfig(7, 8, 2, 3, 24)
        model = variant.GPT(config)
        self.assertFalse(hasattr(model, "position_embedding"))
        for block in model.blocks:
            self.assertIsInstance(block.attention_norm, torch.nn.LayerNorm)
            self.assertIsInstance(block.mlp[1], torch.nn.GELU)
            self.assertIsNotNone(block.attention.qkv.bias)
            self.assertIsNotNone(block.attention.projection.bias)
        tokens = torch.randint(7, (2, 8))
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 7
        torch.testing.assert_close(model(tokens)[:, :4], model(changed)[:, :4], rtol=0, atol=0)
        F.cross_entropy(model(tokens).flatten(0, 1), tokens.flatten()).backward()
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        )
        for length in (0, 9):
            with self.assertRaises(ValueError):
                model(torch.zeros(1, length, dtype=torch.long))
        with self.assertRaises(ValueError):
            variant.ModelConfig(7, n_embd=18, n_head=6)

    def test_count_recipe_loading_and_provenance(self):
        with torch.device("meta"):
            baseline = vanilla.GPT(vanilla.ModelConfig(65))
            model = variant.GPT(variant.ModelConfig(65))
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), 10_795_776)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10_697_472)
        control = load_recipe("recipes/shakespeare-10m.json")
        recipe = load_recipe("recipes/shakespeare-10m-rope.json")
        for key in ("training", "backend", "dataset", "model_overrides"):
            self.assertEqual(getattr(control, key), getattr(recipe, key))
        config = variant.ModelConfig(7, 8, 1, 3, 24)
        model = variant.GPT(config)
        checkpoint = {
            "config": {"model_name": "model_10m_rope", "model": vars(config)},
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
