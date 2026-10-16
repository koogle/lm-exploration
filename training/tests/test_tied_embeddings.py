"""Embedding-sharing correctness checks; no data downloads or training jobs."""

import copy
import hashlib
import io
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from model_100m import model as baseline
from model_100m_tied import model as variant
from torch.nn import functional as F
from training_data.subword import TOKENIZER_SHA256, VOCAB_SIZE, load_frozen

from generate import load_model
from trainer.checkpoints import runtime_provenance, validate_resume
from trainer.recipe import MODELS, Recipe


class TiedEmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = variant.ModelConfig(11, 8, 1, 3, 24)

    def test_shares_one_parameter_and_preserves_other_initial_weights(self):
        control = baseline.GPT(self.config)
        torch.manual_seed(42)
        model = variant.GPT(self.config)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        self.assertEqual(sum(p is model.lm_head.weight for p in model.parameters()), 1)
        for name, parameter in control.named_parameters():
            if name != "lm_head.weight":
                torch.testing.assert_close(parameter, model.state_dict()[name], rtol=0, atol=0)
        optimizer = torch.optim.AdamW(model.parameters())
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(sum(p is model.lm_head.weight for p in parameters), 1)
        self.assertIs(model.double().lm_head.weight, model.token_embedding.weight)

    def test_gradient_sums_input_and_output_roles(self):
        model = variant.GPT(self.config)
        control = baseline.GPT(self.config)
        # Give the untied reference identical VALUES, retaining two independent Parameters.
        control.load_state_dict(model.state_dict())
        self.assertIsNot(control.lm_head.weight, control.token_embedding.weight)
        tokens = torch.randint(self.config.vocab_size, (2, 8))
        targets = (tokens + 1) % self.config.vocab_size
        actual = model(tokens)
        expected = control(tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        F.cross_entropy(actual.flatten(0, 1), targets.flatten()).backward()
        F.cross_entropy(expected.flatten(0, 1), targets.flatten()).backward()
        torch.testing.assert_close(
            model.token_embedding.weight.grad,
            control.token_embedding.weight.grad + control.lm_head.weight.grad,
        )
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        )
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % self.config.vocab_size
        torch.testing.assert_close(model(tokens)[:, :4], model(changed)[:, :4], rtol=0, atol=0)

    def test_checkpoint_roundtrip_preserves_alias_and_logits(self):
        model = variant.GPT(self.config)
        payload = {
            "config": {"model_name": "model_100m_tied", "model": asdict(self.config)},
            "model": model.state_dict(),
        }
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        reloaded = load_model(torch.load(stream, weights_only=True))
        self.assertIs(reloaded.lm_head.weight, reloaded.token_embedding.weight)
        tokens = torch.randint(self.config.vocab_size, (2, 8))
        torch.testing.assert_close(model(tokens), reloaded(tokens), rtol=0, atol=0)
        control = baseline.GPT(self.config)
        old_payload = {
            "config": {"model_name": "model_100m", "model": asdict(self.config)},
            "model": control.state_dict(),
        }
        old = load_model(old_payload)
        self.assertIsNot(old.lm_head.weight, old.token_embedding.weight)

    def test_rejects_cross_architecture_resume_and_keeps_legacy_names(self):
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
        config = {
            "model_name": "model_100m_tied",
            "model": asdict(self.config),
            "tokenizer": {"sha256": TOKENIZER_SHA256},
            "training": training,
        }
        checkpoint = {"config": config, "dataset_fingerprint": {"sha256": "fixture"}, "step": 2}
        fingerprint = checkpoint["dataset_fingerprint"]
        validate_resume(checkpoint, config, fingerprint, 3)
        different = copy.deepcopy(config)
        different["model_name"] = "model_100m"
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_resume(checkpoint, different, fingerprint, 3)
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_resume({**checkpoint, "config": different}, config, fingerprint, 3)
        legacy = copy.deepcopy(different)
        legacy["model_name"] = "100m"
        validate_resume({**checkpoint, "config": legacy}, different, fingerprint, 3)
        del legacy["model_name"]
        validate_resume({**checkpoint, "config": legacy}, different, fingerprint, 3)

    def test_frozen_tokenizer_forward_backward_and_parameter_counts(self):
        tokenizer = load_frozen()
        self.assertEqual(tokenizer.get_vocab_size(), VOCAB_SIZE)
        self.assertEqual(VOCAB_SIZE, 32768)
        ids = tokenizer.encode("def square(x):\n    return x ** 2  # α²\n").ids
        # Also exercise the special IDs and the highest valid vocabulary index.
        tokens = torch.tensor([[0, 1, 2, 3, VOCAB_SIZE - 1, *ids]])
        config = variant.ModelConfig(VOCAB_SIZE, tokens.size(1), 1, 3, 24)
        model = variant.GPT(config)
        logits = model(tokens)
        self.assertEqual(logits.shape, (1, tokens.size(1), VOCAB_SIZE))
        F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()).backward()
        self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all())
        with torch.device("meta"):
            original = baseline.GPT(baseline.ModelConfig(VOCAB_SIZE))
            tied = variant.GPT(variant.ModelConfig(VOCAB_SIZE))
        original_count = sum(p.numel() for p in original.parameters())
        tied_count = sum(p.numel() for p in tied.parameters())
        self.assertEqual(original_count, 149_555_456)
        self.assertEqual(tied_count, 124_389_632)
        self.assertEqual(original_count - tied_count, VOCAB_SIZE * 768)
        self.assertEqual(asdict(original.config), asdict(tied.config))

    def test_registration_and_inherited_source_provenance(self):
        self.assertIs(MODELS["100m_tied"], variant)
        recipe = Recipe(model="100m_tied", dataset="tiny-shakespeare", training={})
        self.assertEqual(recipe.build_model_config(11), baseline.ModelConfig(11))
        sources = runtime_provenance(
            "cpu", variant.__file__, model_dependencies=variant.MODEL_DEPENDENCIES
        )["source_hashes"]
        for name, path in variant.MODEL_DEPENDENCIES.items():
            self.assertEqual(sources[name], hashlib.sha256(Path(path).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
