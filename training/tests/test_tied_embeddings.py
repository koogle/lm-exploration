"""Embedding-sharing correctness checks; no data downloads or training jobs."""

import copy
import io
import unittest

import torch
from model_100m import model as variant
from torch.nn import functional as F
from training_data.subword import VOCAB_SIZE, load_frozen


class TiedEmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = variant.ModelConfig(11, 8, 1, 3, 24)

    def test_default_shares_one_parameter_and_preserves_alias_on_conversion(self):
        model = variant.GPT(self.config)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        self.assertEqual(sum(p is model.lm_head.weight for p in model.parameters()), 1)
        optimizer = torch.optim.AdamW(model.parameters())
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(sum(p is model.lm_head.weight for p in parameters), 1)
        self.assertIs(model.double().lm_head.weight, model.token_embedding.weight)

    def test_gradient_sums_input_and_output_roles(self):
        model = variant.GPT(self.config)
        # An untied mathematical reference exists only inside this test.
        control = copy.deepcopy(model)
        control.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
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
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        reloaded = variant.GPT(self.config)
        reloaded.load_state_dict(torch.load(stream, weights_only=True))
        self.assertIs(reloaded.lm_head.weight, reloaded.token_embedding.weight)
        tokens = torch.randint(self.config.vocab_size, (2, 8))
        torch.testing.assert_close(model(tokens), reloaded(tokens), rtol=0, atol=0)

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
            tied = variant.GPT(variant.ModelConfig(VOCAB_SIZE))
        tied_count = sum(p.numel() for p in tied.parameters())
        self.assertEqual(tied_count, 124_389_632)
        # state_dict names both uses; parameters() counts the shared matrix once.
        named_tensor_count = sum(p.numel() for p in tied.state_dict().values())
        self.assertEqual(named_tensor_count - tied_count, VOCAB_SIZE * 768)


if __name__ == "__main__":
    unittest.main()
