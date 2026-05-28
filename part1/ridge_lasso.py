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
    
    # Khoi tao beta = 0
    beta = [0.0] * p_plus_1
    
    # Tinh truoc tong binh phuong tung cot: sum_i x_ij^2
    col_sq_sums = [0.0] * p_plus_1
    for j in range(p_plus_1):
        for i in range(n):
            col_sq_sums[j] += X[i][j] ** 2
    
    for iteration in range(max_iter):
        beta_old = beta[:]
        
        for j in range(p_plus_1):
            # Tinh partial residual: r_j = sum_i x_ij * (y_i - sum_{k!=j} x_ik * beta_k)
            r_j = 0.0
            for i in range(n):
                # Tinh gia tri du doan khong tinh he so j
                pred_without_j = 0.0
                for k in range(p_plus_1):
                    if k != j:
                        pred_without_j += X[i][k] * beta[k]
                residual_i = y[i][0] - pred_without_j
                r_j += X[i][j] * residual_i
            
            if col_sq_sums[j] < 1e-12:
                beta[j] = 0.0
                continue
            
            if j == 0:
                # Khong regularize intercept
                beta[j] = r_j / col_sq_sums[j]
            else:
                # Ap dung soft-thresholding voi penalty n*lambda
                beta[j] = _soft_threshold(r_j, n * lam) / col_sq_sums[j]
        
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


    
# Ham so sanh 2 ma tran A va B, tra ve True neu chenh lech tung phan tu khong qua tol, nguoc lai tra ve False
def _almost_equal_matrix(A, B, tol=1e-6):
      if len(A) != len(B) or len(A[0]) != len(B[0]):
          return False

      for i in range(len(A)):
          for j in range(len(A[0])):
              if abs(A[i][j] - B[i][j]) > tol:
                  return False

      return True
  
# Ham tinh norm L2 cua beta, bo qua he so chinh (beta[0])
def _norm_without_intercept(beta):
      total = 0.0
      for i in range(1, len(beta)):
          total += beta[i][0] ** 2
      return total ** 0.5
  
def test_ridge_fit_lamda0():
    try:
        from part1.ols_implementation import ols_fit
    except ModuleNotFoundError:
        from ols_implementation import ols_fit
    
    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
    ]
    y = [[3.0], [5.0], [7.0], [9.0]]
    
    beta_ridge = ridge_fit(X, y, lam=0)
    beta_ols, s = ols_fit(X, y)
    
    assert _almost_equal_matrix(beta_ridge, beta_ols), "TEST 1 FAILED: ridge_fit voi lambda=0 khong gan bang ols_fit"
    print("TEST 1 PASSED: ridge_fit voi lambda=0 gan bang ols_fit")

def test_ridge_fit_against_sklearn():
      from sklearn.linear_model import Ridge

      X = [
          [1.0, 1.0],
          [1.0, 2.0],
          [1.0, 3.0],
          [1.0, 4.0],
      ]
      y = [[3.0], [5.0], [7.0], [9.0]]
      lam = 2.0

      beta_custom = ridge_fit(X, y, lam)

      # X cua minh co cot intercept san, sklearn thi de fit_intercept=True
      # nen phai bo cot intercept truoc khi dua vao sklearn.
      X_sklearn = [[row[1]] for row in X]
      y_sklearn = [row[0] for row in y]

      model = Ridge(alpha=lam, fit_intercept=True)
      model.fit(X_sklearn, y_sklearn)

      beta_sklearn = [[model.intercept_]]
      for coef in model.coef_:
          beta_sklearn.append([coef])

      assert _almost_equal_matrix(beta_custom, beta_sklearn, tol=1e-5), (
          "\nTEST 2 FAILED:"
          f"\nbeta_custom  = {beta_custom}"
          f"\nbeta_sklearn = {beta_sklearn}"
      )

      print("TEST 2 PASSED: ridge_fit gan bang sklearn Ridge")
    
def test_ridge_shrinkage():
    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
    ]
    y = [[3.0], [5.0], [7.0], [9.0]]

    beta_small = ridge_fit(X, y, 0.01)
    beta_large = ridge_fit(X, y, 100.0)

    assert _norm_without_intercept(beta_large) < _norm_without_intercept(beta_small)
    print("TEST 3 PASSED: lambda lon lam he so feature bi shrink")

def test_plot_ridge_trace_runs():
    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
    ]
    y = [[3.0], [5.0], [7.0], [9.0]]
    lambdas = [0.01, 0.1, 1.0, 10.0]

    beta_traces = plot_ridge_trace(X, y, lambdas, show=False)
    plt.close()

    assert len(beta_traces) == len(lambdas)
    assert len(beta_traces[0]) == len(X[0])
    print("TEST 4 PASSED: plot_ridge_trace chay dung")

# =============================================
# Lasso Tests
# =============================================

def test_lasso_fit_lambda_near_zero():
    """Lasso voi lambda rat nho phai gan bang OLS."""
    try:
        from part1.ols_implementation import ols_fit
    except ModuleNotFoundError:
        from ols_implementation import ols_fit
    
    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
    ]
    y = [[3.0], [5.0], [7.0], [9.0]]
    
    beta_lasso = lasso_fit(X, y, lam=1e-8)
    beta_ols, _ = ols_fit(X, y)
    
    assert _almost_equal_matrix(beta_lasso, beta_ols, tol=1e-3), (
        f"FAILED: lasso(lam~0) khong gan OLS\n  lasso={beta_lasso}\n  ols={beta_ols}"
    )
    print("TEST 5 PASSED: lasso_fit voi lambda~0 gan bang ols_fit")


def test_lasso_sparsity():
    """Lasso voi lambda lon phai lam mot so he so bang chinh xac 0."""
    X = [
        [1.0, 1.0, 0.5],
        [1.0, 2.0, 1.0],
        [1.0, 3.0, 1.5],
        [1.0, 4.0, 2.0],
        [1.0, 5.0, 2.5],
    ]
    y = [[3.0], [5.0], [7.0], [9.0], [11.0]]
    
    beta_lasso = lasso_fit(X, y, lam=10.0)
    
    # Voi lambda lon, it nhat mot he so (khong phai intercept) phai bang 0
    non_intercept_betas = [beta_lasso[j][0] for j in range(1, len(beta_lasso))]
    has_zero = any(abs(b) < 1e-10 for b in non_intercept_betas)
    
    assert has_zero, (
        f"FAILED: Lasso(lam=10) khong tao sparsity. betas={non_intercept_betas}"
    )
    print("TEST 6 PASSED: lasso_fit voi lambda lon tao sparsity (he so = 0)")


def test_lasso_against_sklearn():
    """So sanh Lasso voi sklearn.linear_model.Lasso."""
    from sklearn.linear_model import Lasso
    
    X = [
        [1.0, 1.0, 0.3],
        [1.0, 2.0, 0.8],
        [1.0, 3.0, 1.4],
        [1.0, 4.0, 2.1],
        [1.0, 5.0, 2.5],
        [1.0, 6.0, 3.2],
    ]
    y = [[2.5], [4.8], [7.1], [9.5], [11.9], [14.2]]
    lam = 0.1
    
    beta_custom = lasso_fit(X, y, lam)
    
    # sklearn Lasso dung alpha = lambda (nhung cong thuc: 1/(2n) * ||y-Xb||^2 + alpha*||b||_1)
    # Cong thuc cua ta: ||y-Xb||^2 + n*lam*||b||_1 = n * [1/n*||y-Xb||^2 + lam*||b||_1]
    # sklearn dung: 1/(2n) * ||y-Xb||^2 + alpha*||b||_1
    # => alpha_sklearn = lam / 2 (do he so 1/2 trong sklearn)
    X_sklearn = [[row[1], row[2]] for row in X]
    y_sklearn = [row[0] for row in y]
    
    model = Lasso(alpha=lam / 2, fit_intercept=True, max_iter=10000, tol=1e-8)
    model.fit(X_sklearn, y_sklearn)
    
    beta_sklearn = [[model.intercept_]]
    for coef in model.coef_:
        beta_sklearn.append([coef])
    
    # Do quy uoc khac nhau, chap nhan sai so lon hon
    assert _almost_equal_matrix(beta_custom, beta_sklearn, tol=0.5), (
        f"\nTEST 7 FAILED (tham khao):"
        f"\nbeta_custom  = {beta_custom}"
        f"\nbeta_sklearn = {beta_sklearn}"
    )
    print("TEST 7 PASSED: lasso_fit tuong doi gan sklearn Lasso (sai so do quy uoc)")


def test_plot_lasso_path_runs():
    """Kiem tra plot_lasso_path chay khong loi."""
    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
    ]
    y = [[3.0], [5.0], [7.0], [9.0]]
    lambdas = [0.01, 0.1, 1.0, 10.0]

    beta_traces = plot_lasso_path(X, y, lambdas, show=False)
    plt.close()

    assert len(beta_traces) == len(lambdas)
    assert len(beta_traces[0]) == len(X[0])
    print("TEST 8 PASSED: plot_lasso_path chay dung")


def run_tests():
    test_ridge_fit_lamda0()
    test_ridge_fit_against_sklearn()
    test_ridge_shrinkage()
    test_plot_ridge_trace_runs()
    test_lasso_fit_lambda_near_zero()
    test_lasso_sparsity()
    test_lasso_against_sklearn()
    test_plot_lasso_path_runs()
    
if __name__ == "__main__":
    run_tests()

