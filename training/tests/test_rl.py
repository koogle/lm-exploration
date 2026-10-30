import json
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from training_data import rl_trajectories

from trainer.rl.algorithms import ALGORITHMS, make
from trainer.rl.batch import Batch, Tokens
from trainer.rl.off_policy.awr import AWR
from trainer.rl.off_policy.ppo import PPO
from trainer.rl.off_policy.sources import TrajectoryFile
from trainer.rl.on_policy.grpo import GRPO
from trainer.rl.rollouts import Scheduler
from trainer.rl.settings import Settings


def tokens(logits, targets, advantage, old_logp=None):
    logp = logits.log_softmax(-1).gather(-1, targets[..., None])[..., 0]
    mask = torch.ones(targets.shape, dtype=torch.bool)
    old = logp.detach() if old_logp is None else old_logp
    return Tokens(logits, targets, mask, logp, old, advantage, torch.ones(len(targets)))


def rows(tasks):
    return [{"id": str(i), "task": task} for i, task in enumerate(tasks)]


class AlgorithmTests(unittest.TestCase):
    def test_algorithms_are_named_by_their_folder(self):
        self.assertEqual(sorted(ALGORITHMS), ["off_policy/awr", "off_policy/ppo", "on_policy/grpo"])
        self.assertEqual(make({"name": "off_policy/ppo", "clip": 0.1}), PPO(clip=0.1))
        for spec in ({"name": "grpo"}, {"name": "on_policy/grpo", "clip": 0.2}):
            with self.subTest(spec=spec), self.assertRaises((ValueError, TypeError)):
                make(spec)

    def test_group_advantages_subtract_the_group_mean(self):
        batch = Batch(rows("math" for _ in range(6)), [], [], [1, 0, 0, 1, 1, 1], [], group=3)
        for got, expected in zip(
            GRPO().advantages(batch), [2 / 3, -1 / 3, -1 / 3, 0, 0, 0], strict=True
        ):
            self.assertAlmostEqual(got, expected)
        scaled = PPO(scale_by_std=True).advantages(batch)
        self.assertAlmostEqual(scaled[0], (2 / 3) / (1 / 3) ** 0.5, places=4)

    def test_grpo_is_the_plain_policy_gradient(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 5, requires_grad=True)
        micro = tokens(logits, torch.tensor([[1, 4, 0], [2, 2, 3]]), torch.tensor([0.5, -1.0]))
        loss, _ = GRPO().loss(micro)
        (grad,) = torch.autograd.grad(loss, logits, retain_graph=True)
        (expected,) = torch.autograd.grad(-(micro.advantage[:, None] * micro.logp).sum(), logits)
        torch.testing.assert_close(grad, expected)

    def test_ppo_weights_by_the_ratio_and_clips_it(self):
        logits = torch.zeros(1, 3, 4, requires_grad=True)
        targets = torch.tensor([[0, 1, 2]])
        # Ratios e^0.5 (clipped for a positive advantage), e^0.1 and 1.
        old = torch.log(torch.tensor([[0.25, 0.25, 0.25]])) - torch.tensor([[0.5, 0.1, 0.0]])
        loss, stats = PPO().loss(tokens(logits, targets, torch.tensor([1.0]), old))
        (grad,) = torch.autograd.grad(loss, logits)
        onehot = torch.nn.functional.one_hot(targets, 4).float()
        expected = -torch.tensor([0.0, 0.1, 0.0]).exp()[:, None] * (onehot[0] - 0.25)
        expected[0] = 0
        torch.testing.assert_close(grad[0], expected)
        self.assertEqual(stats["clipped_tokens"], 1)

    def test_awr_weights_answers_against_their_task_mean(self):
        batch = Batch(rows(["code", "code", "math", "math"]), [], [], [1.0, 0.0, 0.0, 0.0], [])
        self.assertEqual(AWR().advantages(batch), [0.5, -0.5, 0.0, 0.0])
        logits = torch.zeros(3, 1, 4, requires_grad=True)
        micro = tokens(logits, torch.tensor([[0], [0], [0]]), torch.tensor([0.5, -0.5, 30.0]))
        loss, stats = AWR(beta=0.5, max_weight=10.0).loss(micro)
        (grad,) = torch.autograd.grad(loss, logits)
        weights = torch.tensor([1.0, -1.0]).exp().tolist() + [10.0]
        for row, weight in enumerate(weights):
            self.assertAlmostEqual(grad[row, 0, 0].item(), -weight * 0.75, places=5)
        self.assertAlmostEqual(stats["awr_weight"], sum(weights), places=4)


class SettingsTests(unittest.TestCase):
    def test_sections_fill_defaults(self):
        settings = Settings.from_dict(
            {"rollouts": {"math_share": 1.0}, "optimizer": {"lr": 1e-6, "warmup_steps": 10}}
        )
        self.assertEqual(settings.optimizer.learning_rate(4), 5e-7)
        self.assertEqual(settings.rollouts.token_limit("code"), 384)
        self.assertEqual(settings.algorithm, GRPO())
        self.assertEqual(asdict(settings)["trajectories"]["source"], "sampler")

    def test_algorithms_accept_only_their_sources(self):
        reuse = {"source": "reuse", "updates_per_batch": 4}
        teacher = {"source": "file", "data": "teacher-trajectories-01"}
        valid = [
            ("off_policy/ppo", reuse),
            ("off_policy/awr", reuse),
            ("off_policy/awr", teacher),
        ]
        invalid = [
            ("on_policy/grpo", reuse),
            ("off_policy/ppo", teacher),
            ("off_policy/ppo", {}),
        ]
        for name, trajectories in valid:
            Settings.from_dict({"algorithm": {"name": name}, "trajectories": trajectories})
        for name, trajectories in invalid:
            with self.subTest(name=name, trajectories=trajectories), self.assertRaises(ValueError):
                Settings.from_dict({"algorithm": {"name": name}, "trajectories": trajectories})

    def test_invalid_settings(self):
        for values in (
            {"rollouts": {"samples_per_prompt": 1}},
            {"rollouts": {"math_share": 2}},
            {"replay_weight": 0.1, "replay": True},
            {"optimizer": {"warmup_steps": 300}},
            {"trajectories": {"source": "reuse"}},
            {"trajectories": {"source": "file"}},
            {"trajectories": {"source": "sampler", "data": "x"}},
            {"rollout": {}},
        ):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                Settings.from_dict(values)


class SourceTests(unittest.TestCase):
    def test_scheduler_rests_prompts_solved_by_all_or_none_twice(self):
        pool = rows("math" for _ in range(4))
        scheduler = Scheduler(pool, rest=10, rng=random.Random(0))
        scheduler.update(pool[0], 0.0, 0)
        self.assertEqual(scheduler.resting(1), 0)
        scheduler.update(pool[0], 0.0, 1)
        scheduler.update(pool[1], 1.0, 1)
        scheduler.update(pool[1], 0.5, 2)
        self.assertEqual(scheduler.resting(2), 1)
        drawn = {r["id"] for _ in range(20) for r in scheduler.draw("math", 3, 2)}
        self.assertNotIn("0", drawn)
        self.assertEqual(scheduler.resting(11), 0)

    def test_trajectory_file_visits_every_answer_once_per_pass(self):
        written = [
            {**row, "completion_ids": [7, int(row["id"])], "reward": float(i % 2)}
            for i, row in enumerate(rows("code" for _ in range(5)))
        ]
        source = TrajectoryFile(written, 2, random.Random(0), stop=2)
        first = [source.batch(step) for step in range(2)]
        seen = [row["id"] for batch in first for row in batch.rows]
        self.assertEqual(len(set(seen)), 4)
        self.assertEqual(first[0].finished, [True, True])
        self.assertEqual(source.statistics(1), {"trajectory_passes": 1})
        source.batch(2)
        self.assertEqual(source.statistics(2), {"trajectory_passes": 2})

    def test_trajectories_load_training_rows(self):
        good = {"id": "a", "task": "code", "split": "train", "completion": "x", "reward": 1.0}
        held_out = {**good, "id": "b", "split": "validation"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "trajectories.jsonl")
            path.write_text("".join(json.dumps(r) + "\n" for r in (good, held_out)))
            loaded, digest = rl_trajectories.load(path)
            self.assertEqual([r["id"] for r in loaded], ["a"])
            self.assertEqual(len(digest), 64)
            path.write_text(json.dumps({**good, "reward": 0.5}) + "\n")
            with self.assertRaises(ValueError):
                rl_trajectories.load(path)

    def test_teacher_answers_become_rewarded_trajectories(self):
        sft = {
            "id": "p1",
            "split": "train",
            "question": "Add two numbers.",
            "declaration": "def add(a, b):",
            "tests": "assert add(1, 2) == 3",
            "source": "test",
            "license": "MIT",
        }
        answers = [
            "```python\ndef add(a, b):\n    return a + b\n```",
            "```python\ndef add(a, b):\n    return a - b\n```",
            "no code",
        ]

        class Sandbox:
            def run(self, pairs):
                return [{"passed": "a + b" in code} for code, _ in pairs]

        built = rl_trajectories.from_teacher(
            [sft, {**sft, "split": "validation"}], [answers, answers], Sandbox(), "teacher@1"
        )
        self.assertEqual([r["reward"] for r in built], [1.0, 0.0])
        self.assertEqual(built[0]["task"], "code")
        self.assertTrue(built[0]["completion"].startswith("def add"))


if __name__ == "__main__":
    unittest.main()
