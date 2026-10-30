"""StableReinforce (`trainer/rl/off_policy/stable_reinforce.py`) on any algorithm."""

import unittest

import torch

from trainer.rl.batch import Batch, Tokens
from trainer.rl.off_policy.awr import AWR
from trainer.rl.off_policy.stable_reinforce import (
    StableReinforce,
    StableReinforceSettings,
    sampler_kl,
)
from trainer.rl.on_policy.grpo import GRPO
from trainer.rl.settings import Settings

PROMPT = [5, 6]


def tokens_for(logits, targets):
    logits = logits.clone().requires_grad_(True)
    logp = logits.log_softmax(-1).gather(-1, targets[..., None])[..., 0]
    mask = torch.ones(targets.shape, dtype=torch.bool)
    return Tokens(logits, targets, mask, logp, logp.detach(), torch.ones(1), torch.ones(1))


def batch_for(targets, sampler_logprobs, top_ids, top_logprobs):
    """One answer `targets[0]` after a two-token prompt: its targets start at position 1,
    so the micro-batch below holds the prompt's last position and then the answer."""
    picked = sampler_logprobs.gather(-1, targets[..., None])[0, 1:, 0]
    records = [(picked, top_ids[0, 1:], top_logprobs[0, 1:])]
    rows = [{"prompt_ids": PROMPT, "task": "math"}]
    return Batch(rows, [targets[0, 1:].tolist()], [False], [1.0], sampler=records)


def logits_grad(logits, targets, sampler=None, top_k=None, algorithm=None):
    """Gradient on the answer logits with advantage 1 on every answer token, through the
    hook when `sampler` (the sampler's log-probabilities) is given."""
    tokens = tokens_for(logits, targets)
    tokens.mask[0, 0] = False
    penalty = 0.0
    if sampler is not None:
        top, ids = sampler.topk(top_k, -1)
        hook = StableReinforce(StableReinforceSettings(top_k=top_k))
        penalty = hook.apply(tokens, batch_for(targets, sampler, ids, top), [0])
    loss, _ = (algorithm or GRPO()).loss(tokens)
    (loss + penalty).backward()
    return tokens.logits.grad[0, 1:]


class ScoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(1, 4, 10)
        self.targets = torch.tensor([[3, 2, 7, 0]])
        self.onehot = torch.nn.functional.one_hot(self.targets[0, 1:], 10).float()
        self.p = self.logits.softmax(-1)[0, 1:]

    def test_plain_gradient_is_p_minus_onehot(self):
        torch.testing.assert_close(logits_grad(self.logits, self.targets), self.p - self.onehot)

    def test_centering_with_the_learner_distribution_changes_nothing(self):
        plain = logits_grad(self.logits, self.targets)
        learner = torch.log_softmax(self.logits, -1)
        for k in (10, 4):
            with self.subTest(k=k):
                torch.testing.assert_close(
                    logits_grad(self.logits, self.targets, learner, k), plain, atol=1e-6, rtol=0
                )

    def test_centered_gradient_is_q_minus_onehot(self):
        sampler = torch.log_softmax(self.logits + torch.randn(1, 4, 10), -1)
        torch.testing.assert_close(
            logits_grad(self.logits, self.targets, sampler, 10),
            sampler.exp()[0, 1:] - self.onehot,
            atol=1e-6,
            rtol=0,
        )

    def test_top_k_tail_follows_the_learner(self):
        sampler = torch.log_softmax(self.logits + torch.randn(1, 4, 10), -1)
        top, ids = (t[0, 1:] for t in sampler.topk(3, -1))
        estimate = torch.zeros_like(self.p).scatter(-1, ids, top.exp())
        outside = 1 - torch.zeros_like(self.p).scatter(-1, ids, 1.0)
        tail = (1 - top.exp().sum(-1, keepdim=True)) * self.p * outside
        estimate = estimate + tail / (self.p * outside).sum(-1, keepdim=True)
        torch.testing.assert_close(
            logits_grad(self.logits, self.targets, sampler, 3),
            estimate - self.onehot,
            atol=1e-6,
            rtol=0,
        )

    def test_any_algorithm_pushes_up_the_centered_score(self):
        # AWR with advantage 1 weighs every token by exp(1 / beta), capped at max_weight.
        sampler = torch.log_softmax(self.logits + torch.randn(1, 4, 10), -1)
        awr = AWR(beta=1.0)
        centered = logits_grad(self.logits, self.targets, sampler, 10, awr)
        torch.testing.assert_close(
            centered, (sampler.exp()[0, 1:] - self.onehot) * torch.e, atol=1e-5, rtol=0
        )

    def test_sampler_kl_and_mismatch(self):
        logp = torch.log(torch.tensor([[0.2, 0.5]]))
        mask = torch.ones(1, 2)
        self.assertEqual(sampler_kl(logp, logp, mask).abs().sum().item(), 0)
        self.assertGreater(sampler_kl(logp, logp - 0.3, mask).sum().item(), 0)
        sampler = torch.log_softmax(self.logits + 0.5 * torch.randn(1, 4, 10), -1)
        tokens = tokens_for(self.logits, self.targets)
        tokens.mask[0, 0] = False
        hook = StableReinforce(StableReinforceSettings(top_k=0, sampler_kl_coef=0.1))
        top, ids = sampler.topk(2, -1)
        penalty = hook.apply(tokens, batch_for(self.targets, sampler, ids, top), [0])
        self.assertGreater(float(penalty), 0)
        self.assertIs(tokens.score, tokens.logp)
        picked = sampler.gather(-1, self.targets[..., None])[0, 1:, 0]
        expected = (picked - tokens.logp[0, 1:].detach()).mean().item()
        self.assertAlmostEqual(hook.statistics()["sampler_mismatch"], expected, places=5)
        self.assertEqual(hook.statistics(), {})

    def test_settings(self):
        self.assertIsNone(StableReinforce.attach(Settings()))
        settings = Settings.from_dict({"stable_reinforce": {}})
        self.assertEqual(StableReinforce.attach(settings).top_k, 32)
        for values in (
            {"stable_reinforce": {}, "rollouts": {"temperature": 0.7}},
            {"stable_reinforce": {"top_k": -1}},
            {"stable_reinforce": {"sampler_kl_coef": -1.0}},
            {
                "stable_reinforce": {"top_k": 0, "sampler_kl_coef": 0.1},
                "algorithm": {"name": "off_policy/awr"},
                "trajectories": {"source": "file", "data": "teacher"},
            },
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Settings.from_dict(values)
        # The sampler KL alone does not need temperature 1.
        Settings.from_dict({"stable_reinforce": {"top_k": 0}, "rollouts": {"temperature": 0.7}})


if __name__ == "__main__":
    unittest.main()
