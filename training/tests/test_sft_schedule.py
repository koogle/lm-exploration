import unittest

from trainer.sft import Settings


class ScheduleTests(unittest.TestCase):
    def test_cosine_warms_up_and_decays_to_floor(self):
        settings = Settings(
            steps=100, lr=1e-4, schedule="cosine", warmup_steps=10, min_lr_ratio=0.1
        )
        self.assertAlmostEqual(settings.learning_rate(0), 1e-5)
        self.assertAlmostEqual(settings.learning_rate(9), 1e-4)
        self.assertAlmostEqual(settings.learning_rate(10), 1e-4)
        self.assertAlmostEqual(settings.learning_rate(55), 0.55e-4)
        self.assertAlmostEqual(settings.learning_rate(100), 1e-5)
        self.assertEqual(Settings(lr=3e-5).learning_rate(7), 3e-5)

    def test_rejects_inconsistent_schedules(self):
        for values in (
            {"warmup_steps": 5},
            {"schedule": "cosine"},
            {"schedule": "cosine", "min_lr_ratio": 0},
            {"schedule": "linear"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Settings(**values)


if __name__ == "__main__":
    unittest.main()
