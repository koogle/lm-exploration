"""Multi-token prediction in the hybrid family: alignment, masking and inference."""

import dataclasses
import unittest

import torch
from model_hybrid import documents, model
from torch.nn import functional as F

from tests.test_hybrid_backbone import tiny_config

EOS = tiny_config().eos_id


class MTPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        config = tiny_config(n_layer=2, layer_pattern=("kda", "full"), mtp=True)
        self.lm = model.HybridLM(config).double()
        # Two packed documents: [BOS a b c EOS] [BOS d e ...], BOS = 1 never a target.
        self.inputs = torch.tensor([[1, 10, 11, 12, EOS, 1, 13, 14, 15]])
        following = torch.tensor([[10, 11, 12, EOS, 1, 13, 14, 15, 16]])
        self.labels = following.masked_fill(following == 1, -100)

    def test_mtp_targets_skip_document_boundaries(self):
        targets = documents.mtp_labels(self.inputs, self.labels, EOS)
        # Position t predicts x_(t+2): BOS->11, 10->12, 11->EOS, then the boundary.
        self.assertEqual(targets.tolist(), [[11, 12, EOS, -100, -100, 14, 15, 16, -100]])

    def test_loss_alignment_with_explicit_module(self):
        lm, inputs, labels = self.lm, self.inputs, self.labels
        targets = documents.mtp_labels(inputs, labels, EOS)
        losses = lm(inputs, labels, targets)
        x, rope, mask = lm.trunk(inputs)
        next_tokens = torch.cat((inputs[:, 1:], labels[:, -1:]), 1)
        future = lm.mtp(x, lm.token_embedding(next_tokens), rope, mask)
        logits = F.linear(future, lm.output_weight)[0]
        valid = targets[0] != -100
        expected = F.cross_entropy(logits[valid], targets[0, valid], reduction="sum")
        torch.testing.assert_close(losses["mtp_ce"], expected)
        self.assertEqual(losses["mtp_count"].item(), 6)
        self.assertEqual(losses["count"].item(), 8)
        torch.testing.assert_close(
            losses["mtp_loss"], expected + 1e-4 * logits[valid].logsumexp(-1).square().sum()
        )
        (losses["loss"] + 0.3 * losses["mtp_loss"]).backward()
        for name, parameter in lm.named_parameters():
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_mtp_never_sees_its_target(self):
        x, rope, mask = self.lm.trunk(self.inputs)
        next_tokens = torch.cat((self.inputs[:, 1:], self.labels[:, -1:]), 1)

        def future(tokens):
            return self.lm.mtp(x, self.lm.token_embedding(tokens), rope, mask)

        changed = next_tokens.clone()
        changed[0, 6] = 30  # x_7, the target of position 5
        torch.testing.assert_close(future(next_tokens)[0, :6], future(changed)[0, :6])

    def test_mtp_is_off_for_inference(self):
        logits = self.lm(self.inputs)
        trunk_only = model.HybridLM(dataclasses.replace(self.lm.config, mtp=False)).double()
        state = {k: v for k, v in self.lm.state_dict().items() if not k.startswith("mtp.")}
        trunk_only.load_state_dict(state)
        torch.testing.assert_close(trunk_only(self.inputs), logits, rtol=0, atol=0)
        losses = self.lm(self.inputs, self.labels)
        self.assertNotIn("mtp_loss", losses)
        with self.assertRaises(ValueError):
            trunk_only(self.inputs, self.labels, mtp_labels=self.labels)

    def test_weight_schedule(self):
        config = self.lm.config
        self.assertAlmostEqual(model.mtp_weight(config, 0.0), 0.3)
        self.assertAlmostEqual(model.mtp_weight(config, 0.5), 0.2)
        self.assertAlmostEqual(model.mtp_weight(config, 1.0), 0.1)
        self.assertAlmostEqual(model.mtp_weight(config, 2.0), 0.1)


if __name__ == "__main__":
    unittest.main()
