"""The Triton softmax-gradient kernel under Triton's CPU interpreter."""

import os
import subprocess
import sys
import textwrap
import unittest

SCRIPT = textwrap.dedent(
    """
    import torch
    from model_hybrid.softmax_kernel import softmax_gradient_

    torch.manual_seed(0)
    logits = torch.randn(4, 5000) * 3
    targets = torch.tensor([7, -100, 4999, 0])
    z_loss = 1e-2
    reference = logits.double()
    lse = reference.logsumexp(-1)
    expected = (reference - lse[:, None]).exp() * (1 + 2 * z_loss * lse)[:, None]
    expected[torch.arange(4), targets.clamp_min(0)] -= 1
    expected[1] = 0
    actual = logits.clone()
    got_lse, got_target = softmax_gradient_(actual, targets, z_loss)
    torch.testing.assert_close(got_lse.double(), lse, rtol=0, atol=1e-5)
    torch.testing.assert_close(got_target, logits[torch.arange(4), targets.clamp_min(0)])
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=1e-6)
    print("ok")
    """
)


class SoftmaxKernelTests(unittest.TestCase):
    def test_gradient_and_statistics_in_interpreter(self):
        env = {**os.environ, "TRITON_INTERPRET": "1"}
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
