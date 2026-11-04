"""The protected edit must leave control-subspace activations unchanged."""

import unittest

import torch

from .round2 import basis, perpendicular


class ProtectedDirectionTest(unittest.TestCase):
    def test_update_annihilates_protected_inputs(self):
        torch.manual_seed(89)
        controls = torch.randn(8, 32)
        protected = basis(controls, 8)
        contrast = torch.randn(32)
        right = perpendicular(contrast, protected)
        right /= right @ contrast
        update = torch.randn(12, 1) @ right[None, :]
        torch.testing.assert_close(controls @ update.T, torch.zeros(8, 12), atol=2e-6, rtol=0)
        torch.testing.assert_close(right @ contrast, torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
