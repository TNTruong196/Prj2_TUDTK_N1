import math
import unittest

import matplotlib
matplotlib.use("Agg")

from part1.ols_implementation import ols_fit
from part1.residual_analysis import residual_plots


class TestResidualPlots(unittest.TestCase):
    def setUp(self):
        self.X = [
            [1.0, 2.0, 1.5],
            [1.0, 3.0, 2.1],
            [1.0, 4.0, 2.8],
            [1.0, 5.0, 4.3],
            [1.0, 6.0, 5.2],
        ]
        self.y = [
            [2.6],
            [4.2],
            [6.0],
            [8.7],
            [10.4],
        ]
        self.beta_hat, _ = ols_fit(self.X, self.y)

    def test_residual_plots_returns_axes(self):
        fig, axes = residual_plots(self.X, self.y, self.beta_hat, show=False)
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 2)
        self.assertEqual(len(axes[0]), 2)
        self.assertEqual(len(axes[1]), 2)

    def test_residual_plots_handles_noise(self):
        y_noisy = [[row[0] + 0.1] for row in self.y]
        fig, axes = residual_plots(self.X, y_noisy, self.beta_hat, show=False)
        self.assertTrue(hasattr(fig, "savefig"))
        self.assertTrue(all(hasattr(ax, "plot") for ax in axes.flatten()))


if __name__ == "__main__":
    unittest.main()
