import unittest

import torch
from model_hybrid.gated_delta import configure, scalar_reference
from model_hybrid.model import HybridConfig, HybridLM


class ScalarDeltaTests(unittest.TestCase):
    def test_document_reset_and_scalar_decay(self):
        torch.manual_seed(4)
        q, k, v = [torch.randn(1, 7, 2, 4) for _ in range(3)]
        g = torch.randn(1, 7, 2, 1, requires_grad=True)
        beta = torch.randn(1, 7, 2)
        rate, bias = torch.zeros(2), torch.full((2,), -2.0)
        pos = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])
        full = scalar_reference(q, k, v, g, beta, rate, bias, pos)
        tail = scalar_reference(
            q[:, 3:], k[:, 3:], v[:, 3:], g[:, 3:], beta[:, 3:], rate, bias, pos[:, 3:]
        )
        torch.testing.assert_close(full[:, 3:], tail, rtol=0, atol=0)
        (gradient,) = torch.autograd.grad(full[:, 3:].sum(), g)
        self.assertEqual(gradient[:, :3].abs().max().item(), 0)
        self.assertGreater(gradient[:, 3:].abs().max().item(), 0)

    def test_model_backward_and_common_weights(self):
        config = HybridConfig(
            vocab_size=33,
            eos_id=2,
            block_size=16,
            n_layer=2,
            n_embd=16,
            n_head=2,
            n_kv_head=1,
            head_dim=8,
            n_kda_head=2,
            mlp_hidden=32,
            layer_pattern=("kda", "full"),
        )
        model = HybridLM(config)
        original = model.blocks[0].mixer.q_proj.weight.detach().clone()
        configure(model)
        torch.testing.assert_close(model.blocks[0].mixer.q_proj.weight, original)
        self.assertEqual(model.blocks[0].mixer.f_proj.out_features, 2)
        tokens = torch.randint(3, 33, (2, 12))
        labels = tokens.roll(-1, 1)
        output = model(tokens, labels, labels)
        (output["loss"] + 0.3 * output["mtp_loss"]).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
