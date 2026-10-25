"""Consolidated 10M architecture and shared formulas."""

import unittest

import test_100m_modern as shared_tests
import torch
from model_10m import model


class Modern10MTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = model.ModelConfig(11, 8, 1, 3, 24, eos_id=2)

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
    test_documents_are_isolated_and_positions_restart = (
        shared_tests.Modern100MTests.test_documents_are_isolated_and_positions_restart
    )
    test_document_attention_matches_explicit_attention_per_document = (
        shared_tests.Modern100MTests.test_document_attention_matches_explicit_attention_per_document
    )

    def test_modern_classes_are_standalone(self):
        self.assertEqual(model.ModelConfig.__bases__, (object,))
        for cls in (model.Attention, model.Block, model.GPT):
            self.assertEqual(cls.__bases__, (torch.nn.Module,))

    def test_defaults_count_tying_and_checkpoint_roundtrip(self):
        config = model.ModelConfig(65, eos_id=0)
        self.assertEqual((config.n_layer, config.n_embd, config.n_head), (6, 384, 6))
        with torch.device("meta"):
            instance = model.GPT(config)
        self.assertEqual(sum(p.numel() for p in instance.parameters()), 10_670_592)
        self.assertFalse(hasattr(instance, "position_embedding"))
        self.assertEqual(instance.blocks[0].mlp.hidden_width, 1024)
        instance = model.GPT(self.config)
        restored = model.GPT(self.config)
        restored.load_state_dict(instance.state_dict())
        self.assertIs(restored.lm_head.weight, restored.token_embedding.weight)
        tokens = torch.randint(11, (2, 8))
        torch.testing.assert_close(instance(tokens), restored(tokens), rtol=0, atol=0)
        instance(tokens).sum().backward()
        self.assertTrue(torch.isfinite(instance.token_embedding.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
