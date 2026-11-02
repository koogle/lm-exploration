"""`agent_sft_worker.balanced_steps`: every trajectory once, steps of equal work."""

import unittest

import numpy as np

try:
    from modal_scripts import agent_sft_worker
except ImportError:  # torch is not installed
    agent_sft_worker = None


@unittest.skipIf(agent_sft_worker is None, "torch is not installed")
class BalancedStepsTests(unittest.TestCase):
    def test_every_position_once_and_balanced(self):
        rng = np.random.default_rng(0)
        lengths = rng.integers(1000, 40960, 16 * 130 + 5)
        steps = agent_sft_worker.balanced_steps(lengths, 8, 16, 64, 48000)
        self.assertEqual(len(steps), 130)  # the trailing partial step is dropped
        positions = sorted(p for step in steps for rank in step for p in rank)
        self.assertEqual(positions, list(range(16 * 130)))
        cost = lambda p: agent_sft_worker.step_cost(int(lengths[p]), 48000)
        loads = np.array([[sum(cost(p) for p in rank) for rank in step] for step in steps])
        self.assertTrue(all(len(rank) for step in steps for rank in step))
        self.assertGreater(loads.mean(1).sum() / loads.max(1).sum(), 0.9)

    def test_deterministic(self):
        lengths = np.arange(1, 16 * 10 + 1)
        first = agent_sft_worker.balanced_steps(lengths, 8, 16, 4, 48000)
        self.assertEqual(first, agent_sft_worker.balanced_steps(lengths, 8, 16, 4, 48000))


if __name__ == "__main__":
    unittest.main()
