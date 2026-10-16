"""Modern 100M block formulas and compatibility; no training jobs."""

import copy
import unittest
from dataclasses import asdict

import torch
from model_10m import model as vanilla
from model_100m import model
from torch.nn import functional as F

from generate import load_model
from trainer.checkpoints import validate_resume
from trainer.recipe import Recipe


def complex_rotary(x, offset=0):
    pairs = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
    frequencies = 10000.0 ** (-torch.arange(0, x.size(-1), 2).double() / x.size(-1))
    angles = torch.outer(torch.arange(offset, offset + x.size(-2)).double(), frequencies)
    return torch.view_as_real(pairs * torch.polar(torch.ones_like(angles), angles)).flatten(-2)


class Modern100MTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = model.ModelConfig(11, 8, 1, 3, 24)

    def test_modern_defaults_and_parameter_matched_feedforward(self):
        config = model.ModelConfig(32768)
        self.assertEqual(
            (config.n_layer, config.n_embd, config.n_head, config.block_size), (14, 768, 12, 256)
        )
        self.assertEqual(config.architecture_version, 2)
        with torch.device("meta"):
            instance = model.GPT(config)
        self.assertFalse(hasattr(instance, "position_embedding"))
        norms = [m for m in instance.modules() if isinstance(m, model.RMSNorm)]
        self.assertEqual(len(norms), 29)
        self.assertTrue(all(m.eps == 1e-5 for m in norms))
        self.assertFalse(any(isinstance(m, torch.nn.LayerNorm) for m in instance.modules()))
        for block in instance.blocks:
            self.assertIsInstance(block.attention, model.Attention)
            self.assertIsInstance(block.mlp, model.SwiGLU)
            self.assertEqual(block.mlp.hidden_width, 2048)
            matrices = sum(p.numel() for p in block.mlp.parameters() if p.ndim == 2)
            self.assertEqual(matrices, 8 * 768**2)
            self.assertIsNotNone(block.attention.qkv.bias)
            self.assertIsNotNone(block.mlp.gate.bias)

    def test_rmsnorm_outputs_and_gradients_against_pytorch(self):
        norm = model.RMSNorm(8).double()
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
        for dtype in (torch.float16, torch.bfloat16):
            zero = torch.zeros(2, 8, dtype=dtype, requires_grad=True)
            out = model.RMSNorm(8)(zero)
            self.assertEqual(out.dtype, dtype)
            out.sum().backward()
            self.assertTrue(torch.isfinite(zero.grad).all())

    def test_rope_rotation_and_attention_against_independent_reference(self):
        x = torch.randn(1, 2, 3, 8, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(model.rotary(x, 7), complex_rotary(x, 7))
        self.assertTrue(torch.autograd.gradcheck(model.rotary, (x,)))
        torch.testing.assert_close(model.rotary(x).square().sum(-1), x.square().sum(-1))
        key = torch.randn_like(x)
        torch.testing.assert_close(
            model.rotary(x) @ model.rotary(key).transpose(-2, -1),
            model.rotary(x, 17) @ model.rotary(key, 17).transpose(-2, -1),
        )
        attention = model.Attention(self.config).double()
        inputs = torch.randn(2, 5, 24, dtype=torch.float64, requires_grad=True)
        reference_x = inputs.detach().clone().requires_grad_()
        actual = attention(inputs)
        query, key, value = attention.qkv(reference_x).chunk(3, -1)
        query, key, value = [t.view(2, 5, 3, 8).transpose(1, 2) for t in (query, key, value)]
        attended = F.scaled_dot_product_attention(
            complex_rotary(query), complex_rotary(key), value, is_causal=True
        )
        expected = attention.projection(attended.transpose(1, 2).reshape(2, 5, 24))
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, (inputs, *attention.parameters()), upstream)
        expected_grad = torch.autograd.grad(
            expected, (reference_x, *attention.parameters()), upstream
        )
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b)

    def test_swiglu_output_and_parameter_gradients(self):
        layer = model.SwiGLU(24).double()
        x = torch.randn(2, 3, 24, dtype=torch.float64, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        actual = layer(x)
        gate = layer.gate(reference_x)
        expected = layer.down(gate * torch.sigmoid(gate) * layer.up(reference_x))
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, (x, *layer.parameters()), upstream)
        expected_grad = torch.autograd.grad(expected, (reference_x, *layer.parameters()), upstream)
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b)

    def test_combined_model_causality_gradients_and_context_bounds(self):
        instance = model.GPT(self.config)
        tokens = torch.randint(11, (2, 8))
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 11
        torch.testing.assert_close(
            instance(tokens)[:, :4], instance(changed)[:, :4], rtol=0, atol=0
        )
        F.cross_entropy(instance(tokens).flatten(0, 1), tokens.flatten()).backward()
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in instance.parameters())
        )
        for length in (0, 9):
            with self.assertRaises(ValueError):
                instance(torch.zeros(1, length, dtype=torch.long))
        for version in (0, 3, True):
            with self.assertRaises(ValueError):
                model.ModelConfig(11, architecture_version=version)
        with self.assertRaises(ValueError):
            model.ModelConfig(11, n_embd=18, n_head=6)
        self.assertEqual(
            model.ModelConfig(11, n_embd=18, n_head=6, architecture_version=1).n_embd, 18
        )

    def test_old_checkpoints_keep_legacy_blocks_with_exact_logits(self):
        settings = asdict(self.config)
        del settings["architecture_version"]
        tokens = torch.randint(11, (1, 8))
        original = vanilla.GPT(vanilla.ModelConfig(**settings))
        for name in (None, "100m", "model_100m", "model_100m_tied"):
            config = {"model": settings}
            if name is not None:
                config["model_name"] = name
            reference = copy.deepcopy(original)
            if name == "model_100m_tied":
                reference.lm_head.weight = reference.token_embedding.weight
            restored = load_model({"config": config, "model": reference.state_dict()})
            self.assertEqual(restored.config.architecture_version, 1)
            self.assertIsInstance(restored.final_norm, torch.nn.LayerNorm)
            self.assertTrue(hasattr(restored, "position_embedding"))
            torch.testing.assert_close(restored(tokens), reference(tokens), rtol=0, atol=0)
        with torch.device("meta"):
            old = model.GPT(model.ModelConfig(32768, architecture_version=1))
        self.assertEqual(sum(p.numel() for p in old.parameters()), 149_760_000)

    def test_recipe_version_and_resume_compatibility(self):
        recipe = Recipe(model="100m", dataset="tiny-shakespeare", training={})
        self.assertEqual(recipe.build_model_config(11).architecture_version, 2)
        legacy_recipe = recipe.model_copy(update={"model_overrides": {"architecture_version": 1}})
        self.assertEqual(legacy_recipe.build_model_config(11).architecture_version, 1)
        training = {
            "batch_size": 2,
            "lr": 0.001,
            "seed": 42,
            "device": "cuda",
            "dtype": "float32",
            "optimizer": "AdamW",
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "gradient_clip_norm": 1.0,
        }
        settings = asdict(self.config)
        settings["architecture_version"] = 1
        current = {
            "model_name": "model_100m",
            "model": settings,
            "tokenizer": {"type": "fixture"},
            "training": training,
        }
        old = copy.deepcopy(current)
        del old["model"]["architecture_version"]
        checkpoint = {"config": old, "step": 1, "dataset_fingerprint": {"sha256": "fixture"}}
        validate_resume(checkpoint, current, checkpoint["dataset_fingerprint"], 2)
        modern = copy.deepcopy(current)
        modern["model"]["architecture_version"] = 2
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_resume(checkpoint, modern, checkpoint["dataset_fingerprint"], 2)


if __name__ == "__main__":
    unittest.main()
