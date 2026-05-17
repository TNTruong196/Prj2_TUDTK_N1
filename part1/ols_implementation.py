from matrix_helper import *

# Chu y: Vector duoc bieu dien bang 2D list

# X: Ma tran thiet ke kich thuoc nx(p+1) voi cot dau toan 1
# y: Vector kich thuoc nx1 chua cac gia tri quan sat thuc te
# Tra ve: Nghiem OLS va uoc luong phuong sai nhieu
def ols_fit(X, y):
    if not is_matrix(X) or not is_matrix(y):
        raise ValueError("X va y phai la ma tran")
        
    n = len(X)
    p_plus_1 = len(X[0])
    
    if len(y) != n or len(y[0]) != 1:
        raise ValueError("y phai la ma tran nx1")

    # Tinh Beta: (X^T * X)^-1 * X^T * y
    X_t = mat_trans(X)
    X_t_X = mat_mul(X_t, X)
    X_t_X_inv = mat_inverse(X_t_X)
    beta = mat_mul(mat_mul(X_t_X_inv, X_t), y)
    
    # Tinh phan du: e = y - X * beta
    y_hat = mat_mul(X, beta)
    residuals = mat_sub(y, y_hat)
    
    # Tinh RSS
    rss = sum(row[0] ** 2 for row in residuals)
    
    # Tinh sigma^2
    df = n - p_plus_1
    if df <= 0:
        raise ValueError("Khong du bac tu do de tinh phuong sai")
        
    sigma_sqr = rss / df
    
    return beta, sigma_sqr


# X: Ma tran thiet ke kich thuoc nx(p+1)
# Tra ve: Ma tran chieu (hat matrix)
def hat_matrix(X):
    if not is_matrix(X):
        raise ValueError("X phai la ma tran")
    
    # Cong thuc: H = X * (X^T * X)^-1 * X^T
    X_t = mat_trans(X)
    X_t_X = mat_mul(X_t, X)
    X_t_X_inv = mat_inverse(X_t_X)
    
    temp = mat_mul(X, X_t_X_inv)
    H = mat_mul(temp, X_t)
    
    return H