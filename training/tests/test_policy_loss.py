import unittest

import torch

from trainer.policy_loss import sampler_kl, surrogate


def logits_grad(logits, targets, top_ids=None, top_logprobs=None):
    logits = logits.clone().requires_grad_(True)
    mask = torch.ones(targets.shape)
    loss, _ = surrogate(logits, targets, mask, torch.ones(1), top_ids, top_logprobs)
    loss.backward()
    return logits.grad


class PolicyLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(1, 3, 10)
        self.targets = torch.tensor([[2, 7, 0]])
        self.onehot = torch.nn.functional.one_hot(self.targets, 10).float()

    def test_reinforce_gradient_is_p_minus_onehot(self):
        p = self.logits.softmax(-1)
        torch.testing.assert_close(logits_grad(self.logits, self.targets), p - self.onehot)

    def test_centering_with_the_learner_distribution_changes_nothing(self):
        plain = logits_grad(self.logits, self.targets)
        for k in (10, 4):
            top, ids = torch.log_softmax(self.logits, -1).topk(k, -1)
            with self.subTest(k=k):
                torch.testing.assert_close(
                    logits_grad(self.logits, self.targets, ids, top), plain, atol=1e-6, rtol=0
                )

    def test_centered_gradient_is_q_minus_onehot(self):
        sampler = torch.log_softmax(self.logits + torch.randn(1, 3, 10), -1)
        top, ids = sampler.topk(10, -1)
        torch.testing.assert_close(
            logits_grad(self.logits, self.targets, ids, top),
            sampler.exp() - self.onehot,
            atol=1e-6,
            rtol=0,
        )

    def test_top_k_tail_follows_the_learner(self):
        sampler = torch.log_softmax(self.logits + torch.randn(1, 3, 10), -1)
        top, ids = sampler.topk(3, -1)
        p = self.logits.softmax(-1)
        estimate = torch.zeros_like(p).scatter(-1, ids, top.exp())
        outside = 1 - torch.zeros_like(p).scatter(-1, ids, 1.0)
        tail = (1 - top.exp().sum(-1, keepdim=True)) * p * outside
        estimate = estimate + tail / (p * outside).sum(-1, keepdim=True)
        torch.testing.assert_close(
            logits_grad(self.logits, self.targets, ids, top),
            estimate - self.onehot,
            atol=1e-6,
            rtol=0,
        )

    def test_sampler_kl_is_zero_when_engines_agree(self):
        logp = torch.log(torch.tensor([[0.2, 0.5]]))
        mask = torch.ones(1, 2)
        self.assertEqual(sampler_kl(logp, logp, mask).abs().sum().item(), 0)
        self.assertGreater(sampler_kl(logp, logp - 0.3, mask).sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
