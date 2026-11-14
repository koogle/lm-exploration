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
        initial_frozen = gc.get_freeze_count()
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
        self.assertEqual(gc.get_freeze_count(), initial_frozen)

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


class FP8AccountingTests(unittest.TestCase):
    def test_real_model_partition_and_high_precision_weight_gradient(self):
        from pathlib import Path

        import torch
        from model_hybrid.float8_training import eligible
        from model_hybrid.model import HybridLM
        from model_hybrid.precision_metrics import fp8_flops_per_token
        from model_hybrid.train import RunConfig, flops_per_token, model_config_of

        config = model_config_of(
            RunConfig.load(Path(__file__).parents[1] / "hybrid/configs/kda-960m-100b.json")
        )
        with torch.device("meta"):
            lm = HybridLM(config)
        lm._fp8_linear_names = tuple(
            name for name, module in lm.named_modules() if eligible(module, name)
        )
        lm._fp8_recipe = "tensorwise"
        fp8 = fp8_flops_per_token(lm, config)
        self.assertEqual(fp8, 6 * 707002368)
        self.assertLess(fp8, flops_per_token(config, lm, 1000))
        lm._fp8_recipe = "rowwise_with_gw_hp"
        self.assertEqual(fp8_flops_per_token(lm, config), 4 * 707002368)
        lm._fp8_loss_mode = "logits_dx"
        head = 4 * (1 + config.mtp_loss_fraction) * lm.output_weight.numel()
        self.assertEqual(fp8_flops_per_token(lm, config), 4 * 707002368 + head)
