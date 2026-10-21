import unittest

from distillation.evaluate import select_windows


class Data:
    def __init__(self):
        self.rows = iter(
            [("a", [1], [2])] * 12 + [("b", [3], [4])] * 3 + [("c", [5], [6])]
        )
        self.state = {}

    def next(self):
        key, x, y = next(self.rows)
        self.state = {"hash": key}
        return x, y


class EvaluationTests(unittest.TestCase):
    def test_document_and_window_caps(self):
        selected = select_windows(Data(), max_documents=2, windows_per_document=2)
        self.assertEqual([x[0] for x in selected], ["a", "a", "b", "b"])
        self.assertEqual(selected, select_windows(Data(), 2, 2))

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            select_windows(Data(), 0, 2)
