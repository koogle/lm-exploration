"""RL rank processes on CPU (gloo): collectives and the launching process's services."""

import os
import unittest

import torch

from trainer.rl.ranks import Ranks, Services, connect, run_ranks, serve


class Sandbox:
    def run(self, pairs):
        return [{"passed": solution == tests} for solution, tests in pairs]


def collectives(ranks, address, key):
    services = connect(address, key)
    mine = ranks.shard(list(range(6)))
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, float(ranks.rank + 1))
    ranks.average_gradients(model)
    checked = services.run([(str(ranks.rank), "0"), ("x", "x")])
    services.commit()
    return {
        "mine": mine,
        "unshard": ranks.unshard([10 * x for x in mine]),
        "gather": ranks.gather([ranks.rank]),
        "sum": ranks.sum(ranks.rank + 1),
        # Rank 0 trained nothing and recorded no clipping statistic.
        "sum_values": ranks.sum_values(
            {"tokens": 2} if ranks.main else {"tokens": 3, "clipped": 1}
        ),
        "after": ranks.gather([ranks.rank]),
        "broadcast": ranks.broadcast(f"from {ranks.rank}"),
        "grad": model.weight.grad.tolist(),
        "checked": ranks.gather([r["passed"] for r in checked]),
        "stop": services.should_stop(),
    }


class RanksTests(unittest.TestCase):
    def test_single_process_is_the_identity(self):
        ranks = Ranks()
        self.assertEqual(ranks.shard([1, 2]), [1, 2])
        self.assertEqual((ranks.sum(3), ranks.broadcast("x")), (3, "x"))
        self.assertEqual(ranks.sum_values({"tokens": 2}), {"tokens": 2})
        with self.assertRaisesRegex(ValueError, "split"):
            Ranks(0, 2).shard([1, 2, 3])

    @unittest.skipIf(torch.cuda.is_available(), "the CPU path uses gloo")
    def test_two_ranks_share_work_and_the_launching_process_services(self):
        os.environ["MASTER_PORT"] = "29613"
        commits = []
        services = Services(Sandbox(), lambda: commits.append(1), lambda: False)
        result = run_ranks(2, collectives, *serve(services))
        self.assertEqual(result["mine"], [0, 2, 4])
        self.assertEqual(result["unshard"], [0, 10, 20, 30, 40, 50])
        self.assertEqual((result["gather"], result["sum"]), ([0, 1], 3))
        self.assertEqual(result["sum_values"], {"tokens": 5, "clipped": 1})
        self.assertEqual(result["after"], [0, 1])
        self.assertEqual(result["broadcast"], "from 0")
        self.assertEqual(result["grad"], [[1.5, 1.5]])
        self.assertEqual(result["checked"], [True, True, False, True])
        self.assertEqual((len(commits), result["stop"]), (2, False))


if __name__ == "__main__":
    unittest.main()
