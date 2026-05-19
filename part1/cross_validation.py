import random

try:
    from part1.matrix_helper import *
    from part1.ols_implementation import ols_fit
except ModuleNotFoundError:
    from matrix_helper import *
    from ols_implementation import ols_fit

SEED = 42

def ols_beta_fit(X_train, y_train):
    beta, _ = ols_fit(X_train, y_train)
    return beta

def kfold_cv(X, y, k, fit_func = None):
    # fit_func phai tra ve beta dang 2D list de dung duoc voi mat_mul
    if fit_func is None:
        fit_func = ols_beta_fit
        
    n = len(X)
    
    if k < 2 or k > n:
        raise ValueError("k phai nam trong khoang [2, n]")
        
    
    # Tao chi so cho k-fold CV
    indices = list(range(n))
    
    random.seed(SEED)
    random.shuffle(indices)
    
    fold_size = [n // k] * k    
    for i in range(n % k):
        fold_size[i] += 1
        
    folds = []
    start = 0
    for size in fold_size:
        fold_indices = indices[start:start + size]
        folds.append(fold_indices)
        start += size
        
    fold_mse = []
    
    for i in range(k):
        val_indices = folds[i]
        
        train_indices = []
        for j in range(k):
            if j != i:
                for idx in folds[j]:
                    train_indices.append(idx)
        
        X_train = [X[idx] for idx in train_indices]
        y_train = [y[idx] for idx in train_indices]
        
        X_val = [X[idx] for idx in val_indices]
        y_val = [y[idx] for idx in val_indices]
        
        beta = fit_func(X_train, y_train)
        
        # y^
        y_pred = mat_mul(X_val, beta)
        
        mse = 0.0
        for j in range(len(y_val)):
            error = y_val[j][0] - y_pred[j][0]
            mse += error ** 2
            
        mse /= len(y_val)
        fold_mse.append(mse)
        
    cv_mse = sum(fold_mse) / len(fold_mse)

    return cv_mse, fold_mse

def test_kfold_returns_correct_structure():
      X = [
          [1.0, 1.0],
          [1.0, 2.0],
          [1.0, 3.0],
          [1.0, 4.0],
          [1.0, 5.0],
      ]
      y = [[3.0], [5.0], [7.0], [9.0], [11.0]]

      cv_mse, fold_mses = kfold_cv(X, y, 5)

      assert isinstance(cv_mse, float)
      assert isinstance(fold_mses, list)
      assert len(fold_mses) == 5
      assert all(isinstance(mse, float) for mse in fold_mses)
      assert all(mse >= 0 for mse in fold_mses)

      print("TEST 1 PASSED: kfold_cv tra ve dung cau truc")


def test_kfold_reproducible():
    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
        [1.0, 5.0],
        [1.0, 6.0],
    ]
    y = [[3.0], [5.0], [7.0], [9.0], [11.0], [13.0]]

    result1 = kfold_cv(X, y, 3)
    result2 = kfold_cv(X, y, 3)

    assert result1 == result2

    print("TEST 2 PASSED: kfold_cv reproducible voi seed=42")


def test_kfold_with_custom_fit_func():
    try:
        from part1.ridge_lasso import ridge_fit
    except ModuleNotFoundError:
        from ridge_lasso import ridge_fit

    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
        [1.0, 5.0],
        [1.0, 6.0],
    ]
    y = [[3.0], [5.0], [7.0], [9.0], [11.0], [13.0]]

    fit_func = lambda X_train, y_train: ridge_fit(X_train, y_train, 1.0)

    cv_mse, fold_mses = kfold_cv(X, y, 3, fit_func=fit_func)

    assert isinstance(cv_mse, float)
    assert len(fold_mses) == 3
    assert cv_mse >= 0

    print("TEST 3 PASSED: kfold_cv chay duoc voi custom fit_func Ridge")
    

def run_tests():
    test_kfold_returns_correct_structure()
    test_kfold_reproducible()
    test_kfold_with_custom_fit_func()
    
if __name__ == "__main__":
    run_tests()
    
    
    
    
