import unittest
from types import SimpleNamespace

import torch

from campaign_generate import accepted, sparse
from megakernel.targets import compact_targets


class TargetTests(unittest.TestCase):
    def test_selected_token_outside_topk_keeps_full_normalization(self):
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0], [5.0, 0.0, -1.0, 2.0]])
        selected = torch.tensor([0, 3])
        actual = compact_targets(logits, selected, k=2)
        for i in range(2):
            full = logits[i].log_softmax(-1)
            self.assertAlmostEqual(
                actual["selected_logprobs"][i], full[selected[i]].item(), places=6
            )
            mass = torch.tensor(actual["topk_logprobs"][i]).exp().sum().item()
            self.assertAlmostEqual(mass + actual["tail_mass"][i], 1.0, places=6)
            self.assertGreater(actual["tail_mass"][i], 0.0)

    def test_vllm_selected_target_and_token_alignment(self):
        values = torch.tensor([0.0, 1.0, 2.0]).log_softmax(-1).tolist()
        row = {i: SimpleNamespace(logprob=p) for i, p in enumerate(values)}
        result = sparse([row, row], [0, 2])
        self.assertEqual(result["selected_logprobs"], [values[0], values[2]])
        with self.assertRaises(ValueError):
            sparse([row], [0, 2])

    def test_truncated_or_invalid_examples_are_retained_as_rejections(self):
        text = '{"problem":"p","code":"def solve(): return 1","explanation":"e"}'
        self.assertTrue(accepted(text, "stop"))
        self.assertFalse(accepted(text, "length"))
        self.assertFalse(accepted("not JSON", "stop"))
