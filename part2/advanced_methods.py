"""
Bayesian Linear Regression cho phan nang cao cua Phan 2.

Thuat toan chinh duoc cai dat bang 2D list va cac helper ma tran tu part1.
NumPy/sklearn khong duoc dung trong phan fit/predict chinh.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from part1.matrix_helper import mat_add, mat_identity, mat_inverse, mat_mul, mat_trans
from part1.ols_implementation import ols_fit


def _validate_xy(X, y):
    if not X or not y:
        raise ValueError("X va y khong duoc rong")
    if len(X) != len(y):
        raise ValueError("X va y phai co cung so dong")
    if len(y[0]) != 1:
        raise ValueError("y phai la vector cot dang nx1")

    p = len(X[0])
    for row in X:
        if len(row) != p:
            raise ValueError("Tat ca dong cua X phai co cung so cot")


def _zero_vector(size):
    return [[0.0] for _ in range(size)]


def _diag_precision(size, prior_variance, intercept_prior_variance):
    if prior_variance <= 0:
        raise ValueError("prior_variance phai > 0")
    if intercept_prior_variance <= 0:
        raise ValueError("intercept_prior_variance phai > 0")

    precision = mat_identity(size)
    for i in range(size):
        variance = intercept_prior_variance if i == 0 else prior_variance
        precision[i][i] = 1.0 / variance
    return precision


def _scale_matrix(A, scalar):
    return [[value * scalar for value in row] for row in A]


def bayesian_linear_fit(
    X,
    y,
    sigma2=None,
    prior_mean=None,
    prior_variance=100.0,
    intercept_prior_variance=1e12,
):
    """
    Fit Bayesian Linear Regression voi prior Gaussian doc lap cho cac he so.

    Cong thuc:
        S_n = (S_0^-1 + X^T X / sigma2)^-1
        m_n = S_n (S_0^-1 m_0 + X^T y / sigma2)

    Parameters:
        X: 2D list n x (p+1), da co cot intercept.
        y: 2D list n x 1.
        sigma2: phuong sai nhieu. Neu None, uoc luong bang OLS tren train.
        prior_mean: vector cot (p+1) x 1. Neu None, dung vector 0.
        prior_variance: phuong sai prior cho cac feature coefficients.
        intercept_prior_variance: phuong sai prior cho intercept, de lon de gan
            voi intercept khong bi regularize.

    Returns:
        dict gom posterior_mean, posterior_covariance, sigma2 va prior settings.
    """
    _validate_xy(X, y)

    p_plus_1 = len(X[0])
    if prior_mean is None:
        prior_mean = _zero_vector(p_plus_1)
    elif len(prior_mean) != p_plus_1 or len(prior_mean[0]) != 1:
        raise ValueError("prior_mean phai co kich thuoc (p+1)x1")

    if sigma2 is None:
        _, sigma2 = ols_fit(X, y)
    if sigma2 <= 0:
        raise ValueError("sigma2 phai > 0")

    X_t = mat_trans(X)
    X_t_X = mat_mul(X_t, X)
    X_t_y = mat_mul(X_t, y)

    prior_precision = _diag_precision(
        p_plus_1,
        prior_variance=prior_variance,
        intercept_prior_variance=intercept_prior_variance,
    )

    likelihood_precision = _scale_matrix(X_t_X, 1.0 / sigma2)
    posterior_precision = mat_add(prior_precision, likelihood_precision)
    posterior_covariance = mat_inverse(posterior_precision)

    prior_part = mat_mul(prior_precision, prior_mean)
    likelihood_part = _scale_matrix(X_t_y, 1.0 / sigma2)
    posterior_rhs = mat_add(prior_part, likelihood_part)
    posterior_mean = mat_mul(posterior_covariance, posterior_rhs)

    return {
        "posterior_mean": posterior_mean,
        "posterior_covariance": posterior_covariance,
        "sigma2": sigma2,
        "prior_mean": prior_mean,
        "prior_variance": prior_variance,
        "intercept_prior_variance": intercept_prior_variance,
    }


def bayesian_predict(X, posterior):
    """Du doan y_hat = X @ E[beta | X, y]."""
    beta_mean = posterior["posterior_mean"]
    return mat_mul(X, beta_mean)


def credible_intervals(posterior, level=0.95):
    """
    Tinh credible interval xap xi cho tung he so.

    Mac dinh dung he so 1.96 cho muc 95%, phu hop de bao cao ngan gon trong
    phan nang cao. Cac muc khac dung normal quantile tu scipy neu co.
    """
    if not 0 < level < 1:
        raise ValueError("level phai nam trong (0, 1)")

    if abs(level - 0.95) < 1e-12:
        z_value = 1.96
    else:
        from scipy.stats import norm

        z_value = norm.ppf(0.5 + level / 2.0)

    mean = posterior["posterior_mean"]
    cov = posterior["posterior_covariance"]
    intervals = []
    for i in range(len(mean)):
        se = cov[i][i] ** 0.5
        center = mean[i][0]
        intervals.append((center - z_value * se, center + z_value * se))
    return intervals


def _almost_equal_matrix(A, B, tol=1e-5):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False
    for i in range(len(A)):
        for j in range(len(A[0])):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True


def test_bayesian_weak_prior_close_to_ols():
    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
        [1.0, 5.0],
    ]
    y = [[3.0], [5.0], [7.0], [9.0], [11.0]]

    beta_ols, _ = ols_fit(X, y)
    posterior = bayesian_linear_fit(
        X,
        y,
        sigma2=1.0,
        prior_variance=1e12,
        intercept_prior_variance=1e12,
    )

    assert _almost_equal_matrix(posterior["posterior_mean"], beta_ols, tol=1e-4)
    print("TEST 1 PASSED: weak prior cho posterior mean gan OLS")


def test_bayesian_strong_prior_shrinks_feature_coef():
    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
        [1.0, 5.0],
    ]
    y = [[3.2], [4.9], [7.1], [9.0], [10.8]]

    weak = bayesian_linear_fit(X, y, prior_variance=1e12)
    strong = bayesian_linear_fit(X, y, prior_variance=0.01)

    assert abs(strong["posterior_mean"][1][0]) < abs(weak["posterior_mean"][1][0])
    print("TEST 2 PASSED: prior manh lam he so feature bi shrink")


def test_bayesian_predict_shape():
    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
    ]
    y = [[2.0], [4.0], [6.0], [8.0]]
    posterior = bayesian_linear_fit(X, y, prior_variance=10.0)
    y_hat = bayesian_predict(X, posterior)

    assert len(y_hat) == len(X)
    assert len(y_hat[0]) == 1
    print("TEST 3 PASSED: bayesian_predict tra ve vector cot dung kich thuoc")


def run_tests():
    test_bayesian_weak_prior_close_to_ols()
    test_bayesian_strong_prior_shrinks_feature_coef()
    test_bayesian_predict_shape()


if __name__ == "__main__":
    run_tests()
