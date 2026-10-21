import json
import unittest

import torch

from distillation.alignment import (
    byte_decoder,
    byte_probabilities,
    marginal_kl,
    shared_boundaries,
    vocabulary,
)


class AlignmentTests(unittest.TestCase):
    def test_byte_alphabet_is_bijective(self):
        decoder = byte_decoder()
        self.assertEqual(set(decoder.values()), set(range(256)))
        self.assertEqual(decoder["Ġ"], 32)
        self.assertEqual(decoder["Ċ"], 10)
        self.assertEqual(decoder["Ã"], 195)

    def test_boundary_shift_and_unicode(self):
        student = [None, b"a", b"\xc3", b"\xa9", b"b"]
        teacher = [b"a\xc3\xa9", b"b"]
        self.assertEqual(
            shared_boundaries([0, 1, 2, 3, 4], [0, 1], student, teacher),
            ([3, 4], [0, 1]),
        )
        with self.assertRaises(ValueError):
            shared_boundaries([1], [0], student, teacher)

    def test_marginal_matches_manual_and_backpropagates(self):
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], requires_grad=True)
        probs = byte_probabilities(logits, [65, 65, 66, 256])
        reference = logits.softmax(-1)
        torch.testing.assert_close(probs[0, 65], reference[0, :2].sum())
        torch.testing.assert_close(probs.sum(-1), torch.ones(1))
        loss = marginal_kl(logits, probs.detach(), [65, 65, 66, 256])
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        target = torch.zeros_like(probs)
        target[0, 66] = 1
        marginal_kl(logits, target, [65, 65, 66, 256]).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0, 2].item(), 0)

    def test_vocab_specials_unused_rows_and_multibyte(self):
        spec = {
            "model": {"type": "BPE", "vocab": {"Ġx": 0, "<eos>": 1, "Ã": 2}},
            "decoder": {"type": "ByteLevel"},
            "normalizer": None,
            "added_tokens": [{"id": 1, "special": True}],
        }
        pieces, buckets = vocabulary(json.dumps(spec), 4)
        self.assertEqual(pieces, [b" x", None, b"\xc3", None])
        self.assertEqual(buckets, [32, 256, 195, 256])


if __name__ == "__main__":
    unittest.main()
