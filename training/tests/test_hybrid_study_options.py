import dataclasses
import unittest

import torch
from model_hybrid.model import HybridLM, ShortConvolution

from tests.test_hybrid_backbone import tiny_config


class ShortConvolutionTests(unittest.TestCase):
    def test_identity_at_initialization_and_document_reset(self):
        torch.manual_seed(0)
        conv = ShortConvolution(6, 4)
        x = torch.randn(2, 9, 6)
        positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 0, 1]] * 2)
        torch.testing.assert_close(conv(x, positions), x)
        with torch.no_grad():
            conv.weight.normal_()
        out = conv(x, positions)
        changed = x.clone()
        changed[:, :4] += 1.0  # the first document
        later = conv(changed, positions)
        torch.testing.assert_close(later[:, 4:], out[:, 4:])
        self.assertFalse(torch.allclose(later[:, 3], out[:, 3]))
        # A future token never changes an earlier output.
        changed = x.clone()
        changed[:, 6] += 1.0
        torch.testing.assert_close(conv(changed, positions)[:, :6], out[:, :6])

    def test_model_with_convolutions_starts_as_the_plain_model(self):
        torch.manual_seed(0)
        config = tiny_config()
        plain = HybridLM(config).eval()
        study = HybridLM(dataclasses.replace(config, short_conv_size=4)).eval()
        missing = study.load_state_dict(plain.state_dict(), strict=False)
        self.assertTrue(all("convolution" in key for key in missing.missing_keys))
        self.assertFalse(missing.unexpected_keys)
        tokens = torch.randint(3, config.vocab_size, (2, 16))
        tokens[:, 6] = config.eos_id
        torch.testing.assert_close(study(tokens), plain(tokens), rtol=1e-5, atol=1e-5)
        declared = {id(p) for p, _ in _declared(study)}
        self.assertEqual(declared, {id(p) for p in study.parameters()})

    def test_embedding_norm_trains(self):
        torch.manual_seed(0)
        config = dataclasses.replace(tiny_config(), embedding_norm=True, short_conv_size=4)
        lm = HybridLM(config)
        tokens = torch.randint(3, config.vocab_size, (2, 16))
        losses = lm(tokens, tokens)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIsNotNone(lm.embedding_norm.weight.grad)
        self.assertIsNotNone(lm.blocks[0].mlp_convolution.weight.grad)


def _declared(module):
    for child in module.modules():
        if hasattr(child, "optimizer"):
            yield from child.optimizer()


if __name__ == "__main__":
    unittest.main()
