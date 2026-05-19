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

def run_tests():
    test_ridge_fit_lamda0()
    test_ridge_fit_against_sklearn()
    test_ridge_shrinkage()
    test_plot_ridge_trace_runs()
    
if __name__ == "__main__":
    run_tests()
