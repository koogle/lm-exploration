"""The loss forecast recovers a known curve."""

import unittest

import numpy as np

from publication.loss_forecast import fit, forecast, predict


class LossForecastTests(unittest.TestCase):
    def test_fit_recovers_known_curve(self):
        tokens_b = np.linspace(1, 30, 3000)
        loss = 2.05 + 0.4 * tokens_b**-0.7
        curve = fit(tokens_b, loss, 2.0, 30.0)
        self.assertAlmostEqual(curve["b"], 0.7, delta=0.01)
        self.assertAlmostEqual(float(predict(curve, 85.0)), 2.05 + 0.4 * 85**-0.7, delta=1e-3)

    def test_forecast_ignores_tokens_after_the_cutoff(self):
        rows = [{"tokens": t * 1e9, "loss": 2.0 + 0.5 * t**-0.6} for t in np.linspace(1, 20, 2000)]
        rows += [{"tokens": t * 1e9, "loss": 1.0} for t in np.linspace(20, 40, 2000)]
        result = forecast(rows, until=20.0)
        self.assertEqual([f["window"][0] for f in result["fits"]], [2.0, 5.0, 10.0])
        self.assertAlmostEqual(
            float(predict(result["central"], 85.0)), 2.0 + 0.5 * 85**-0.6, delta=2e-3
        )


if __name__ == "__main__":
    unittest.main()
