import json
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from distillation.shared import take_batch, token_losses, transfer_embeddings


class SharedTests(unittest.TestCase):
    def test_kl_matches_reference_padding_and_teacher_frozen(self):
        torch.manual_seed(1)
        s = torch.randn(2, 3, 7, requires_grad=True)
        t = torch.randn(2, 3, 7, requires_grad=True)
        y = torch.tensor([[1, 2, 3], [4, -100, -100]])
        ce, kl = token_losses(s, t, y)
        keep = y != -100
        expected = F.kl_div(
            s[keep].log_softmax(-1), t.detach()[keep].softmax(-1), reduction="sum"
        )
        torch.testing.assert_close(kl, expected)
        torch.testing.assert_close(
            ce,
            F.cross_entropy(
                s.flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum"
            ),
        )
        (ce + kl).backward()
        self.assertIsNone(t.grad)
        self.assertTrue(torch.equal(s.grad[~keep], torch.zeros_like(s.grad[~keep])))
        self.assertTrue(torch.isfinite(s.grad).all())

    def test_accumulation_uses_global_target_count(self):
        torch.manual_seed(2)
        s = torch.randn(2, 3, 5, requires_grad=True)
        t = torch.randn(2, 3, 5)
        y = torch.tensor([[0, 1, 2], [3, -100, -100]])
        a, b = token_losses(s, t, y)
        ((a + b) / 4).backward()
        expected = s.grad.clone()
        s.grad = None
        for i in range(2):
            a, b = token_losses(s[i : i + 1], t[i : i + 1], y[i : i + 1])
            ((a + b) / 4).backward()
        torch.testing.assert_close(s.grad, expected)

    def test_exact_boundary_cursor(self):
        class Data:
            def __init__(self):
                self.state = {"offset": 0}
                self.iterator = self.blocks()

            def blocks(self):
                tokens = list(range(12))
                for start in range(self.state["offset"], 11, 4):
                    y = tokens[start + 1 : start + 5]
                    x = tokens[start : start + len(y)]
                    self.state = {"offset": start + len(y)}
                    yield x, y

            def next(self):
                return next(self.iterator)

        data = Data()
        first = take_batch(data, 4, 6)
        self.assertEqual(sum(len(y) for _, y in first), 6)
        self.assertEqual(data.state["offset"], 6)
        data.iterator = data.blocks()
        second = take_batch(data, 4, 5)
        self.assertEqual([v for _, y in first + second for v in y], list(range(1, 12)))

    def test_embedding_copy_composition_and_eos(self):
        def spec(vocab, specials):
            return json.dumps(
                {
                    "model": {"type": "BPE", "vocab": vocab},
                    "decoder": {"type": "ByteLevel"},
                    "added_tokens": [{"id": i, "special": True} for i in specials],
                }
            )

        class Old:
            def to_str(self):
                return spec(
                    {"<pad>": 0, "<bos>": 1, "<eos>": 2, "a": 3, "b": 4}, [0, 1, 2]
                )

            def encode(self, text, **kwargs):
                return SimpleNamespace(ids=[{"a": 3, "b": 4}[c] for c in text])

        new = SimpleNamespace(
            backend_tokenizer=SimpleNamespace(
                to_str=lambda: spec({"a": 0, "ab": 1, "<end>": 2}, [2])
            )
        )
        old = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        weight = torch.zeros(4, 3)
        stats = transfer_embeddings(old, Old(), new, weight, 2)
        torch.testing.assert_close(weight[0], old[3])
        torch.testing.assert_close(weight[1], old[3:5].mean(0))
        torch.testing.assert_close(weight[2], old[2])
        self.assertEqual(
            stats, {"exact": 1, "composed": 1, "random_special_or_unused": 2}
        )
