import unittest

from training_data.review import require_data_review


class DataReviewTests(unittest.TestCase):
    def test_reviewed_snapshot_needs_no_override(self):
        require_data_review(True)
        require_data_review(True, True)

    def test_pending_snapshot_requires_explicit_permission(self):
        with self.assertRaisesRegex(ValueError, "pending audits"):
            require_data_review(False)
        require_data_review(False, True)


if __name__ == "__main__":
    unittest.main()
