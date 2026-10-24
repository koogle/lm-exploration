"""Document ids, cu_seqlens and document isolation in packed windows."""

import unittest

import torch
from model_hybrid import documents, model

from tests.test_hybrid_backbone import tiny_config

EOS = 2


class FixedMixture:
    def __init__(self, inputs):
        self.inputs, self.state = inputs, "cursor"

    def next_batch(self, batch_size, device="cpu"):
        return self.inputs[:batch_size].to(device), self.inputs[:batch_size].clone()


class DocumentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_document_ids_positions_and_cu_seqlens(self):
        inputs = torch.tensor([[7, 8, EOS, 1, 9, EOS, 0, 0], [5, 6, 7, 8, 9, 10, 11, 12]])
        ids = documents.document_ids(inputs, EOS)
        self.assertEqual(ids.tolist(), [[0, 0, 0, 1, 1, 1, 2, 2], [0] * 8])
        self.assertEqual(documents.document_positions(ids).tolist()[0], [0, 1, 2, 0, 1, 2, 0, 1])
        self.assertEqual(documents.cu_seqlens(ids).tolist(), [0, 3, 6, 8, 16])
        self.assertEqual(documents.cu_seqlens(ids).dtype, torch.int32)

    def test_loader_wrapper_emits_document_ids(self):
        inputs = torch.tensor([[3, EOS, 4, 5], [6, 7, EOS, 8]])
        loader = documents.DocumentBatches(FixedMixture(inputs), EOS)
        batch = loader.next_batch(2)
        self.assertEqual(batch.documents.tolist(), [[0, 0, 1, 1], [0, 0, 0, 1]])
        self.assertEqual(batch.mtp_labels.tolist(), [[-100, -100, 5, -100], [7, -100, -100, -100]])
        self.assertEqual(loader.state, "cursor")

    def test_document_b_is_unaffected_by_document_a(self):
        lm = model.HybridLM(tiny_config(n_layer=3)).double()
        b = torch.tensor([11, 12, 13, 14, 15])
        first = torch.cat((torch.tensor([20, 21, 22, EOS]), b))[None]
        second = torch.cat((torch.tensor([30, 31, 32, 33, 34, 35, EOS]), b))[None]
        outputs = [lm(x, documents=documents.document_ids(x, EOS)) for x in (first, second)]
        torch.testing.assert_close(outputs[0][0, 4:], outputs[1][0, 7:], rtol=0, atol=1e-12)
        alone = lm(b[None])
        torch.testing.assert_close(outputs[0][0, 4:], alone[0], rtol=0, atol=1e-12)
        # Without document ids, document A leaks into B.
        self.assertFalse(torch.allclose(lm(first)[0, 4:], alone[0]))

    def test_padding_rows_and_gradients(self):
        lm = model.HybridLM(tiny_config())
        inputs = torch.tensor([[4, 5, EOS, 0, 0, 0], [6, EOS, 7, 8, EOS, 9]])
        labels = torch.tensor([[5, EOS, -100, -100, -100, -100], [EOS, 7, 8, EOS, 9, 10]])
        losses = lm(inputs, labels, documents.document_ids(inputs, EOS))
        self.assertEqual(losses["count"].item(), 8)
        losses["loss"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in lm.parameters()))


if __name__ == "__main__":
    unittest.main()
