import matplotlib.pyplot as plt
import math 

try:
    from part1.matrix_helper import *
except ModuleNotFoundError:
    from matrix_helper import *

def ridge_fit(X, y, lam):
    XT = mat_trans(X)
    XTX = mat_mul(XT, X)
    
    # Lay kich thuoc cua ma tran X^T * X
    size = len(XTX)
    I = mat_identity(size)
    I[0][0] = 0  # Bo qua he so chinh 
    
    ridge_matrix = mat_add(XTX, mat_scalar_mul(I, lam))
    ridge_matrix_inv = mat_inverse(ridge_matrix)
    
    XTy = mat_mul(XT, y)
    
    # beta = (X^T X + lambda * I')^-1 X^T y
    beta = mat_mul(ridge_matrix_inv, XTy)
    
    return beta

# Ham ve ridge trace, cho phep nguoi dung truyen vao list lambda tuong ung, neu khong truyen thi se su dung 100 gia tri lambda tu 10^-4 den 10^4
def plot_ridge_trace(X, y, lambdas=None, show=True, feature_names=None, max_features=None):
    if lambdas is None:
        lambdas = [10 ** (-4 + 8 * i / 99) for i in range(100)]

    if not isinstance(lambdas, list) or len(lambdas) == 0:
        raise ValueError("lambdas phai la list cac gia tri lambda hoac None")

    for value in lambdas:
        if value <= 0:
            raise ValueError("Tat ca lambda phai > 0 de ve log10(lambda)")

    beta_traces = []

    for value in lambdas:
        beta = ridge_fit(X, y, value)
        beta_traces.append([row[0] if isinstance(row, list) else row for row in beta])

    log_lambdas = [math.log10(value) for value in lambdas]
    num_coef = len(beta_traces[0]) if beta_traces else 0
    coef_indices = list(range(num_coef))

    if max_features is not None:
        if max_features <= 0:
            raise ValueError("max_features phai la so duong")
        start_idx = 1 if num_coef > 1 else 0
        end_idx = min(num_coef, start_idx + max_features)
        coef_indices = list(range(start_idx, end_idx))

    plt.figure(figsize=(10, 6))

    for j in coef_indices:
        coef_values = [beta[j] for beta in beta_traces]
        if feature_names is not None and j < len(feature_names):
            label = feature_names[j]
        else:
            label = f"beta_{j}"
        plt.plot(log_lambdas, coef_values, label=label)

    plt.title("Ridge Trace")
    plt.xlabel("log10(lambda)")
    plt.ylabel("Coefficient value")
    if len(coef_indices) <= 15:
        plt.legend(loc="best", fontsize=8)
    else:
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()

    return beta_traces

def ridge_trace(X, y, lam=None):
    return plot_ridge_trace(X, y, lam)

# Lasso Regression (L1 Regularization)

def _soft_threshold(z, gamma):
    """Ham soft-threshold: S(z, gamma) = sign(z) * max(|z| - gamma, 0)"""
    if z > gamma:
        return z - gamma
    elif z < -gamma:
        return z + gamma
    else:
        return 0.0


def lasso_fit(X, y, lam, max_iter=1000, tol=1e-6):
    """
    Cai dat Lasso Regression bang Coordinate Descent.
    
    Cong thuc cap nhat cho tung he so j:
        r_j = sum_i x_ij * (y_i - sum_{k!=j} x_ik * beta_k)
        beta_j = soft_threshold(r_j, n*lam) / sum_i x_ij^2
    
    Luu y: Khong regularize intercept (cot 0).
    
    Tham so:
        X: Ma tran thiet ke nx(p+1) voi cot dau toan 1 (intercept)
        y: Vector nx1
        lam: He so regularization lambda >= 0
        max_iter: So vong lap toi da
        tol: Nguong hoi tu
        
    Tra ve: beta dang 2D list (p+1)x1
    """
    if not isinstance(X, list) or not isinstance(y, list):
        raise ValueError("X va y phai la 2D list")
    
    n = len(X)
    p_plus_1 = len(X[0])
    
    if len(y) != n or len(y[0]) != 1:
        raise ValueError("y phai la ma tran nx1")
    
    if lam < 0:
        raise ValueError("lambda phai >= 0")
    
    # Precompute columns of X to avoid slow 2D lookups inside N-loops
    X_cols = mat_trans(X)
    
    # Khoi tao beta = 0
    beta = [0.0] * p_plus_1
    
    # Tinh truoc tong binh phuong tung cot: sum_i x_ij^2
    col_sq_sums = [sum(x**2 for x in col) for col in X_cols]
            
    # Khoi tao vector residuals: r_i = y_i - X_i @ beta
    # Do beta ban dau = 0, residuals = y
    residuals = [y[i][0] for i in range(n)]
    
    for iteration in range(max_iter):
        beta_old = beta[:]
        
        for j in range(p_plus_1):
            if col_sq_sums[j] < 1e-12:
                beta[j] = 0.0
                continue
            
            # Tinh r_j = sum_i x_ij * residuals_i + col_sq_sums[j] * beta[j]
            dot_prod = 0.0
            X_col = X_cols[j]
            for i in range(n):
                dot_prod += X_col[i] * residuals[i]
            r_j = dot_prod + col_sq_sums[j] * beta[j]
            
            beta_old_j = beta[j]
            if j == 0:
                beta[j] = r_j / col_sq_sums[j]
            else:
                beta[j] = _soft_threshold(r_j, n * lam) / col_sq_sums[j]
            
            diff = beta[j] - beta_old_j
            if abs(diff) > 1e-15:
                for i in range(n):
                    residuals[i] -= X_col[i] * diff
        
        # Kiem tra hoi tu: max|beta_new - beta_old| < tol
        max_change = max(abs(beta[j] - beta_old[j]) for j in range(p_plus_1))
        if max_change < tol:
            break
            
    # Chuyen ve 2D list (p+1)x1
    return [[b] for b in beta]


def plot_lasso_path(X, y, lambdas=None, show=True, feature_names=None, max_features=None):
    """Ve Lasso coefficient path theo log10(lambda)."""
    if lambdas is None:
        lambdas = [10 ** (-4 + 8 * i / 99) for i in range(100)]

    if not isinstance(lambdas, list) or len(lambdas) == 0:
        raise ValueError("lambdas phai la list cac gia tri lambda hoac None")

    for value in lambdas:
        if value <= 0:
            raise ValueError("Tat ca lambda phai > 0 de ve log10(lambda)")

    beta_traces = []
    for value in lambdas:
        beta = lasso_fit(X, y, value)
        beta_traces.append([row[0] if isinstance(row, list) else row for row in beta])

    log_lambdas = [math.log10(value) for value in lambdas]
    num_coef = len(beta_traces[0]) if beta_traces else 0
    coef_indices = list(range(num_coef))

    if max_features is not None:
        if max_features <= 0:
            raise ValueError("max_features phai la so duong")
        start_idx = 1 if num_coef > 1 else 0
        end_idx = min(num_coef, start_idx + max_features)
        coef_indices = list(range(start_idx, end_idx))

    plt.figure(figsize=(10, 6))
    for j in coef_indices:
        coef_values = [beta[j] for beta in beta_traces]
        if feature_names is not None and j < len(feature_names):
            label = feature_names[j]
        else:
            label = f"beta_{j}"
        plt.plot(log_lambdas, coef_values, label=label)

    plt.title("Lasso Coefficient Path")
    plt.xlabel("log10(lambda)")
    plt.ylabel("Coefficient value")
    if len(coef_indices) <= 15:
        plt.legend(loc="best", fontsize=8)
    else:
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()

    return beta_traces


    
# Helper functions for tests

def _almost_equal_matrix(A, B, tol=1e-6):
      if len(A) != len(B) or len(A[0]) != len(B[0]):
          return False

      for i in range(len(A)):
          for j in range(len(A[0])):
              if abs(A[i][j] - B[i][j]) > tol:
                  return False

      return True
  
def _norm_without_intercept(beta):
      total = 0.0
      for i in range(1, len(beta)):
          total += beta[i][0] ** 2
      return total ** 0.5

def test_ridge_fit():
    # Test case 1: Ridge fit with lambda = 0 should be close to OLS fit
    try:
        from part1.ols_implementation import ols_fit
    except ModuleNotFoundError:
        from ols_implementation import ols_fit
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
    y = [[3.0], [5.0], [7.0], [9.0]]
    beta_ridge = ridge_fit(X, y, lam=0)
    beta_ols, _ = ols_fit(X, y)
    assert _almost_equal_matrix(beta_ridge, beta_ols)

    # Test case 2: Ridge coefficients should shrink compared to small lambda
    beta_small = ridge_fit(X, y, 0.01)
    beta_large = ridge_fit(X, y, 100.0)
    assert _norm_without_intercept(beta_large) < _norm_without_intercept(beta_small)

def test_plot_ridge_trace():
    # Test case 1: returns correct number of traces and points
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    y = [[3.0], [5.0], [7.0]]
    traces = plot_ridge_trace(X, y, lambdas=[0.1, 1.0], show=False)
    assert len(traces) == 2
    assert len(traces[0]) == 2

    # Test case 2: check if ValueError raised with invalid max_features
    try:
        plot_ridge_trace(X, y, lambdas=[0.1], show=False, max_features=-1)
        assert False, "Should raise ValueError due to negative max_features"
    except ValueError:
        pass

def test_ridge_trace():
    # Test case 1: calling ridge_trace performs plot_ridge_trace and returns results
    X = [[1.0, 1.0], [1.0, 2.0]]
    y = [[3.0], [5.0]]
    traces = ridge_trace(X, y, lam=[0.1, 1.0])
    assert len(traces) == 2

    # Test case 2: handles custom lambdas correctly
    assert len(traces[0]) == len(X[0])

def test__soft_threshold():
    # Test case 1: soft threshold of zero/insufficient value returns 0.0
    assert _soft_threshold(0.5, 1.0) == 0.0
    assert _soft_threshold(-0.5, 1.0) == 0.0

    # Test case 2: soft threshold of large positive/negative values shrinks them
    assert _soft_threshold(2.5, 1.0) == 1.5
    assert _soft_threshold(-2.5, 1.0) == -1.5

def test_lasso_fit():
    # Test case 1: lasso fit with small lambda is close to OLS fit
    try:
        from part1.ols_implementation import ols_fit
    except ModuleNotFoundError:
        from ols_implementation import ols_fit
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    y = [[3.0], [5.0], [7.0]]
    beta_lasso = lasso_fit(X, y, lam=1e-8)
    beta_ols, _ = ols_fit(X, y)
    assert _almost_equal_matrix(beta_lasso, beta_ols, tol=1e-3)

    # Test case 2: lasso fit with high lambda introduces sparsity (coefficient is exactly 0)
    X_sparsity = [[1.0, 1.0, 0.5], [1.0, 2.0, 1.0], [1.0, 3.0, 1.5]]
    y_sparsity = [[3.0], [5.0], [7.0]]
    beta_lasso_sparse = lasso_fit(X_sparsity, y_sparsity, lam=10.0)
    non_intercept_betas = [beta_lasso_sparse[j][0] for j in range(1, len(beta_lasso_sparse))]
    assert any(abs(b) < 1e-10 for b in non_intercept_betas)

def test_plot_lasso_path():
    # Test case 1: returns correct number of traces and points
    X = [[1.0, 1.0], [1.0, 2.0]]
    y = [[3.0], [5.0]]
    traces = plot_lasso_path(X, y, lambdas=[0.1, 1.0], show=False)
    assert len(traces) == 2
    assert len(traces[0]) == 2

    # Test case 2: check if ValueError raised with invalid lambda <= 0
    try:
        plot_lasso_path(X, y, lambdas=[0.0], show=False)
        assert False, "Should raise ValueError due to lambda <= 0"
    except ValueError:
        pass

def test__almost_equal_matrix():
    # Test case 1: returns True for close matrices
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[1.0000001, 2.0], [3.0, 3.9999999]]
    assert _almost_equal_matrix(A, B, tol=1e-5) is True

    # Test case 2: returns False for different shapes or values
    assert _almost_equal_matrix(A, [[1.0]], tol=1e-5) is False
    assert _almost_equal_matrix(A, B, tol=1e-9) is False

def test__norm_without_intercept():
    # Test case 1: ignores first element (intercept)
    beta = [[10.0], [3.0], [4.0]]
    assert abs(_norm_without_intercept(beta) - 5.0) < 1e-9

    # Test case 2: returns 0.0 if only intercept is present
    assert _norm_without_intercept([[10.0]]) == 0.0

def main():
    test_ridge_fit()
    test_plot_ridge_trace()
    test_ridge_trace()
    test__soft_threshold()
    test_lasso_fit()
    test_plot_lasso_path()
    test__almost_equal_matrix()
    test__norm_without_intercept()
    print("All tests passed in ridge_lasso.py!")

if __name__ == "__main__":
    main()

