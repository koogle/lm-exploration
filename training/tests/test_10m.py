"""Consolidated 10M architecture, shared formulas and historical checkpoint checks."""

import copy
import unittest
from dataclasses import asdict
from pathlib import Path

import test_100m_modern as shared_tests
import torch
from model_10m import legacy, model

from generate import load_model
from trainer.checkpoints import validate_resume
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

    def test_defaults_count_tying_and_checkpoint_roundtrip(self):
        config = model.ModelConfig(65)
        self.assertEqual((config.n_layer, config.n_embd, config.n_head), (6, 384, 6))
        self.assertEqual(config.architecture_version, 2)
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

    def test_legacy_checkpoint_logits_and_resume_version_guard(self):
        settings = asdict(self.config)
        del settings["architecture_version"]
        original = legacy.GPT(legacy.ModelConfig(**settings))
        tokens = torch.randint(11, (2, 8))
        for name in ("10m", "model_10m"):
            config = {"model_name": name, "model": settings}
            restored = load_model({"config": config, "model": original.state_dict()})
            self.assertEqual(restored.config.architecture_version, 1)
            torch.testing.assert_close(original(tokens), restored(tokens), rtol=0, atol=0)
        with torch.device("meta"):
            historical = model.GPT(model.ModelConfig(65, architecture_version=1))
        self.assertEqual(sum(p.numel() for p in historical.parameters()), 10_795_776)
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
        old = {"model_name": "model_10m", "model": settings, "tokenizer": {}, "training": training}
        current = copy.deepcopy(old)
        current["model"]["architecture_version"] = 1
        checkpoint = {"config": old, "dataset_fingerprint": {}, "step": 1}
        validate_resume(checkpoint, current, {}, 2)
        current["model"]["architecture_version"] = 2
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_resume(checkpoint, current, {}, 2)

    def test_only_two_model_choices_and_all_recipes_validate(self):
        self.assertEqual(set(MODELS), {"10m", "100m"})
        for path in Path("recipes").glob("*.json"):
            recipe = load_recipe(path)
            self.assertEqual(recipe.build_model_config(65).architecture_version, 2)


if __name__ == "__main__":
    unittest.main()
