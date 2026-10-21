import unittest

from workflows.checks import check_capacity, check_progress, check_resume


class WorkflowTests(unittest.TestCase):
    def test_capacity_is_still_required(self):
        report = {
            "supported_mixture_exposures": 100,
            "sources": {"a": {"validation_loss_tokens": 1}},
        }
        check_capacity(report, 100)
        with self.assertRaises(ValueError):
            check_capacity(report, 101)
        report["sources"]["a"]["validation_loss_tokens"] = 0
        with self.assertRaises(ValueError):
            check_capacity(report, 100)

    def test_stops_and_resume_are_checked(self):
        report = {
            "total_loss_tokens": 95,
            "stop_reason": "token_limit",
            "final_validation": {"weighted_loss": 1.0},
        }
        self.assertTrue(check_progress(report, 80, 100, 10))
        for change in (
            {"total_loss_tokens": 80},
            {"total_loss_tokens": 101},
            {"stop_reason": "source_two_pass_limit"},
            {"final_validation": {"weighted_loss": float("nan")}},
        ):
            with self.assertRaises(ValueError):
                check_progress({**report, **change}, 80, 100, 10)
        check_resume(report, {"initial_validation": report["final_validation"]})
        with self.assertRaises(ValueError):
            check_resume(report, {"initial_validation": {"weighted_loss": 2.0}})


if __name__ == "__main__":
    unittest.main()
