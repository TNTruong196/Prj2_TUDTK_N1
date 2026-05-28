import math

import matplotlib.pyplot as plt
import scipy.stats as stats

try:
    from part1.matrix_helper import is_matrix, mat_mul, mat_sub
    from part1.ols_implementation import hat_matrix
except ModuleNotFoundError:
    from matrix_helper import is_matrix, mat_mul, mat_sub
    from ols_implementation import hat_matrix


# X, y, beta_hat la 2D list
# Tra ve (fig, axes) de notebook co the tai su dung
# Neu show=True thi hien thi do thi ngay

def residual_plots(X, y, beta_hat, show=True):
    if not is_matrix(X) or not is_matrix(y) or not is_matrix(beta_hat):
        raise ValueError("X, y, beta_hat phai la ma tran 2D")

    if len(X) != len(y):
        raise ValueError("X va y phai co cung so dong")

    n = len(X)
    p_plus_1 = len(X[0])

    y_hat = mat_mul(X, beta_hat)
    residuals = mat_sub(y, y_hat)

    rss = sum(row[0] ** 2 for row in residuals)
    df = n - p_plus_1
    if df <= 0:
        raise ValueError("Khong du bac tu do de tinh phuong sai")

    sigma2 = rss / df
    sigma = math.sqrt(sigma2)

    hat = hat_matrix(X)
    leverage = [hat[i][i] for i in range(n)]

    standardized_residuals = []
    cooks_distance = []
    for i in range(n):
        h_ii = leverage[i]
        denom = sigma * math.sqrt(max(1e-12, 1.0 - h_ii))
        r_i = residuals[i][0]
        r_std = r_i / denom
        standardized_residuals.append(r_std)

        cook_denom = max(1e-12, (1.0 - h_ii) ** 2)
        cook = (r_i ** 2 / (p_plus_1 * sigma2)) * (h_ii / cook_denom)
        cooks_distance.append(cook)

    fitted = [row[0] for row in y_hat]
    residual_vals = [row[0] for row in residuals]
    scale_location = [math.sqrt(abs(value)) for value in standardized_residuals]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Residuals vs Fitted
    ax = axes[0][0]
    ax.scatter(fitted, residual_vals, alpha=0.7)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residuals vs Fitted")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")

    # Q-Q plot
    ax = axes[0][1]
    stats.probplot(standardized_residuals, dist="norm", plot=ax)
    ax.set_title("Normal Q-Q")

    # Scale-Location
    ax = axes[1][0]
    ax.scatter(fitted, scale_location, alpha=0.7)
    ax.set_title("Scale-Location")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Sqrt(|Standardized residuals|)")

    # Cook's Distance
    ax = axes[1][1]
    indices = list(range(1, n + 1))
    markerline, stemlines, baseline = ax.stem(indices, cooks_distance)
    markerline.set_markerfacecolor("C3")
    markerline.set_markersize(4)
    baseline.set_linewidth(0.5)
    ax.set_title("Cook's Distance")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Cook's distance")
    threshold = 4 / n if n > 0 else 0
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


# Test

import unittest
import matplotlib
matplotlib.use("Agg")

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
        # Sử dụng hàm ols_fit trực tiếp từ file hiện tại
        try:
            from part1.ols_implementation import ols_fit
        except ModuleNotFoundError:
            from ols_implementation import ols_fit
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
    print("BẮT ĐẦU CHẠY CÁC BÀI TEST CHO RESIDUAL_ANALYSIS...")
    unittest.main()
