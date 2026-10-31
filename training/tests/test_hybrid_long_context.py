"""Long-context evaluation helpers: in-document positions."""

import unittest

import numpy as np
from model_hybrid import train


class LongContextTests(unittest.TestCase):
    def test_positions_restart_after_each_document(self):
        inputs = np.array([[5, 6, 9, 7, 8, 8], [1, 9, 2, 2, 2, 2]])
        expected = [[0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 2, 3]]
        np.testing.assert_array_equal(train.document_positions(inputs, 9), expected)


if __name__ == "__main__":
    unittest.main()
