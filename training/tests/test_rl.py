import json
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from trainer.rl.algorithms import GRPO, Tokens, make
from trainer.rl.rollouts import Scheduler
from trainer.rl.settings import Settings


class AlgorithmTests(unittest.TestCase):
    def test_grpo_advantages_subtract_the_group_mean(self):
        rewards = [1, 0, 0, 1, 1, 1]
        for got, expected in zip(
            GRPO().advantages(rewards, 3), [2 / 3, -1 / 3, -1 / 3, 0, 0, 0], strict=True
        ):
            self.assertAlmostEqual(got, expected)
        scaled = GRPO(scale_by_std=True).advantages(rewards, 3)
        self.assertAlmostEqual(scaled[0], (2 / 3) / (1 / 3) ** 0.5, places=4)
        self.assertEqual(scaled[3:], [0, 0, 0])

    def tokens(self, logits, targets, advantage, old_logp=None):
        logp = logits.log_softmax(-1).gather(-1, targets[..., None])[..., 0]
        mask = torch.ones(targets.shape, dtype=torch.bool)
        old = logp.detach() if old_logp is None else old_logp
        return Tokens(logits, targets, mask, logp, old, advantage, torch.ones(len(targets)))

    def test_on_policy_grpo_is_the_plain_policy_gradient(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 5, requires_grad=True)
        targets = torch.tensor([[1, 4, 0], [2, 2, 3]])
        tokens = self.tokens(logits, targets, torch.tensor([0.5, -1.0]))
        loss, stats = GRPO().loss(tokens)
        (grad,) = torch.autograd.grad(loss, logits, retain_graph=True)
        plain = -(tokens.advantage[:, None] * tokens.logp).sum()
        (expected,) = torch.autograd.grad(plain, logits)
        torch.testing.assert_close(grad, expected)
        self.assertEqual(stats["clipped_tokens"], 0)

    def test_off_policy_grpo_weights_by_the_ratio_and_clips_it(self):
        logits = torch.zeros(1, 3, 4, requires_grad=True)
        targets = torch.tensor([[0, 1, 2]])
        # Ratios e^0.5 (clipped for a positive advantage), e^0.1 and 1.
        old = torch.log(torch.tensor([[0.25, 0.25, 0.25]])) - torch.tensor([[0.5, 0.1, 0.0]])
        tokens = self.tokens(logits, targets, torch.tensor([1.0]), old)
        loss, stats = GRPO(policy="off", updates_per_batch=2).loss(tokens)
        (grad,) = torch.autograd.grad(loss, logits)
        onehot = torch.nn.functional.one_hot(targets, 4).float()
        expected = -torch.tensor([0.0, 0.1, 0.0]).exp()[:, None] * (onehot[0] - 0.25)
        expected[0] = 0
        torch.testing.assert_close(grad[0], expected)
        self.assertEqual(stats["clipped_tokens"], 1)

    def test_policies_and_names_are_checked(self):
        self.assertEqual(make({"name": "grpo", "policy": "off", "updates_per_batch": 4}).clip, 0.2)
        for spec in (
            {"name": "ppo"},
            {"name": "grpo", "policy": "on", "updates_per_batch": 2},
            {"name": "grpo", "policy": "off"},
            {"name": "grpo", "policy": "async"},
        ):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                make(spec)


class SettingsTests(unittest.TestCase):
    def test_sections_fill_defaults(self):
        settings = Settings.from_dict(
            {"rollouts": {"math_share": 1.0}, "optimizer": {"lr": 1e-6, "warmup_steps": 10}}
        )
        self.assertEqual(settings.optimizer.learning_rate(4), 5e-7)
        self.assertEqual(settings.rollouts.token_limit("code"), 384)
        self.assertEqual(settings.algorithm, GRPO())
        self.assertEqual(asdict(settings)["algorithm"]["policy"], "on")

    def test_invalid_settings(self):
        for values in (
            {"rollouts": {"samples_per_prompt": 1}},
            {"rollouts": {"math_share": 2}},
            {"replay_weight": 0.1, "replay": True},
            {"optimizer": {"warmup_steps": 300}},
            {"rollout": {}},
        ):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                Settings.from_dict(values)

    def test_command_prints_resolved_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            recipe = Path(folder, "recipe.json")
            recipe.write_text(json.dumps({"settings": {"algorithm": {"name": "grpo"}}}))
            printed = subprocess.run(
                [sys.executable, "-m", "trainer.rl", "--recipe", str(recipe)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        self.assertEqual(json.loads(printed)["rollouts"]["samples_per_prompt"], 8)


class SchedulerTests(unittest.TestCase):
    def test_scheduler_rests_prompts_solved_by_all_or_none_twice(self):
        rows = [{"id": str(i), "task": "math"} for i in range(4)]
        scheduler = Scheduler(rows, rest=10, rng=random.Random(0))
        scheduler.update(rows[0], 0.0, 0)
        self.assertEqual(scheduler.resting(1), 0)
        scheduler.update(rows[0], 0.0, 1)
        scheduler.update(rows[1], 1.0, 1)
        scheduler.update(rows[1], 0.5, 2)
        self.assertEqual(scheduler.resting(2), 1)
        drawn = {r["id"] for _ in range(20) for r in scheduler.draw("math", 3, 2)}
        self.assertNotIn("0", drawn)
        self.assertEqual(scheduler.resting(11), 0)


if __name__ == "__main__":
    unittest.main()
