import gc
import unittest
import weakref

from model_hybrid.precision_metrics import utilization

from trainer.gc_policy import TrainingGC


class GCPolicyTests(unittest.TestCase):
    def test_new_cycles_collect_and_frozen_graph_is_released_on_exit(self):
        class Node:
            pass

        policy = TrainingGC(2)
        enabled = gc.isenabled()
        try:
            self.assertIsNone(policy.after_step(1))
            old = Node()
            old.cycle = old
            old_ref = weakref.ref(old)
            record = policy.after_step(2)
            self.assertGreater(record["frozen_objects"], 0)
            del old
            new = Node()
            new.cycle = new
            new_ref = weakref.ref(new)
            del new
            gc.collect()
            self.assertIsNone(new_ref())
            self.assertIsNotNone(old_ref())
            self.assertIsNone(policy.after_step(3))
            self.assertEqual(gc.isenabled(), enabled)
        finally:
            policy.close()
        gc.collect()
        self.assertIsNone(old_ref())
        self.assertEqual(gc.get_freeze_count(), 0)

    def test_disabled_and_invalid_policy(self):
        self.assertIsNone(TrainingGC().after_step(5))
        with self.assertRaises(ValueError):
            TrainingGC(-1)


class PrecisionMetricTests(unittest.TestCase):
    def test_precision_ceiling_limits(self):
        bf16 = utilization(989e12, 0, 1, 1)
        self.assertEqual(bf16["mfu"], 1)
        self.assertEqual(bf16["mfu_fp8_peak"], 0.5)
        self.assertEqual(bf16["mfu_mixed_precision"], 1)
        fp8 = utilization(1978e12, 1978e12, 1, 1)
        self.assertEqual(fp8["mfu_mixed_precision"], 1)
        mixed = utilization(989e12, 989e12 / 2, 1, 1)
        self.assertEqual(mixed["mfu_mixed_precision"], 0.75)
        self.assertEqual(mixed["estimated_fp8_flop_fraction"], 0.5)
        self.assertEqual(utilization(989e12, 0, 8, 8), bf16)

    def test_invalid_partition(self):
        with self.assertRaises(ValueError):
            utilization(1, 2, 1, 1)
