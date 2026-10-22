"""Consolidated 10M architecture and shared formulas."""

import json
import unittest
from dataclasses import asdict
from pathlib import Path

import test_100m_modern as shared_tests
import torch
from model_10m import model

from distributed.config import DistributedConfig
from generate import load_model
from trainer.recipe import MODELS, load_recipe


class Modern10MTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = model.ModelConfig(11, 8, 1, 3, 24)

    # Both model sizes use these same implementations; exercise the 10M config too.
    test_rmsnorm_outputs_and_gradients_against_pytorch = (
        shared_tests.Modern100MTests.test_rmsnorm_outputs_and_gradients_against_pytorch
    )
    test_rope_rotation_and_attention_against_independent_reference = (
        shared_tests.Modern100MTests.test_rope_rotation_and_attention_against_independent_reference
    )
    test_swiglu_output_and_parameter_gradients = (
        shared_tests.Modern100MTests.test_swiglu_output_and_parameter_gradients
    )
    test_combined_model_causality_gradients_and_context_bounds = (
        shared_tests.Modern100MTests.test_combined_model_causality_gradients_and_context_bounds
    )

    def test_modern_classes_are_standalone(self):
        self.assertEqual(model.ModelConfig.__bases__, (object,))
        for cls in (model.Attention, model.Block, model.GPT):
            self.assertEqual(cls.__bases__, (torch.nn.Module,))

    def test_defaults_count_tying_and_checkpoint_roundtrip(self):
        config = model.ModelConfig(65)
        self.assertEqual((config.n_layer, config.n_embd, config.n_head), (6, 384, 6))
        with torch.device("meta"):
            instance = model.GPT(config)
        self.assertEqual(sum(p.numel() for p in instance.parameters()), 10_670_592)
        self.assertFalse(hasattr(instance, "position_embedding"))
        self.assertEqual(instance.blocks[0].mlp.hidden_width, 1024)
        instance = model.GPT(self.config)
        restored = load_model(
            {
                "config": {"model_name": "model_10m", "model": asdict(self.config)},
                "model": instance.state_dict(),
            }
        )
        self.assertIs(restored.lm_head.weight, restored.token_embedding.weight)
        tokens = torch.randint(11, (2, 8))
        torch.testing.assert_close(instance(tokens), restored(tokens), rtol=0, atol=0)
        instance(tokens).sum().backward()
        self.assertTrue(torch.isfinite(instance.token_embedding.weight.grad).all())

    def test_only_two_model_choices_and_all_recipes_validate(self):
        self.assertEqual(set(MODELS), {"10m", "100m"})
        for path in Path("workflows/model_configs/historical").glob("*.json"):
            with self.subTest(recipe=path.name):
                settings = json.loads(path.read_text())
                if "dataset" in settings:
                    recipe = load_recipe(path)
                    self.assertIsInstance(recipe.build_model_config(65), model.ModelConfig)
                else:
                    DistributedConfig(**settings)


if __name__ == "__main__":
    unittest.main()
