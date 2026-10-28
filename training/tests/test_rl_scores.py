import unittest

import torch

from trainer.rl.algorithms import GRPO, Tokens
from trainer.rl.scores import sampler_kl


def logits_grad(logits, targets, top_ids=None, top_logprobs=None):
    """Gradient on the logits of on-policy GRPO with advantage 1 on every token."""
    logits = logits.clone().requires_grad_(True)
    logp = logits.log_softmax(-1).gather(-1, targets[..., None])[..., 0]
    mask = torch.ones(targets.shape, dtype=torch.bool)
    tokens = Tokens(logits, targets, mask, logp, logp.detach(), torch.ones(1), torch.ones(1))
    algorithm = GRPO()
    if top_ids is not None:
        algorithm = GRPO(score="centered", centering_top_k=top_ids.size(-1))
        tokens.sampler_top_ids, tokens.sampler_top_logprobs = top_ids, top_logprobs
    loss, _ = algorithm.loss(tokens)
    loss.backward()
    return logits.grad


class ScoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(1, 3, 10)
        self.targets = torch.tensor([[2, 7, 0]])
        self.onehot = torch.nn.functional.one_hot(self.targets, 10).float()

    def test_plain_gradient_is_p_minus_onehot(self):
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

    def test_centering_needs_temperature_one(self):
        from trainer.rl.settings import Settings

        self.assertEqual(GRPO(score="centered").sampler_top_k, 32)
        self.assertEqual(GRPO().sampler_top_k, 0)
        with self.assertRaises(ValueError):
            Settings.from_dict(
                {
                    "algorithm": {"name": "grpo", "score": "centered"},
                    "rollouts": {"temperature": 0.7},
                }
            )


if __name__ == "__main__":
    unittest.main()
