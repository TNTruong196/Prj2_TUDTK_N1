import math
import scipy.stats as stats

try:
    from part1.matrix_helper import *
except ModuleNotFoundError:
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

def model_metrics(y, y_hat, p):
    n = len(y)
    y_bar = sum(row[0] for row in y) / n
    
    rss = sum((y[i][0] - y_hat[i][0])**2 for i in range(n))
    tss = sum((row[0] - y_bar)**2 for row in y)
    
    r_sqr = 1 - (rss / tss)
    adj_r_sqr = 1 - ((rss / (n - p - 1)) / (tss / (n - 1)))
    
    # Kiem dinh F
    # F = [(TSS-RSS)/p] / [RSS/(n-p-1)]
    f_stat = ((tss - rss) / p) / (rss / (n - p - 1))
    f_p_val = 1 - stats.f.cdf(f_stat, p, n - p - 1)
    
    return {
        "RSS": rss,
        "TSS": tss,
        "R_squared": r_sqr,
        "Adj_R_squared": adj_r_sqr,
        "F_statistic": f_stat,
        "F_p_value": f_p_val
    }

def coef_inference(X, y, beta_hat, sigma2):
    n = len(X)
    p = len(X[0]) - 1
    
    X_t = mat_trans(X)
    X_t_X = mat_mul(X_t, X)
    X_t_X_inv = mat_inverse(X_t_X)
    
    se = []
    t_stats = []
    p_values = []
    conf_intervals = []
    
    # t critical value for 95% CI
    t_crit = stats.t.ppf(0.975, n - p - 1)
    
    for i in range(len(beta_hat)):
        # SE = sqrt(sigma2 * C_ii)
        var_beta_i = sigma2 * X_t_X_inv[i][i]
        se_i = math.sqrt(var_beta_i)
        se.append(se_i)
        
        # t-stat = beta / SE
        t_i = beta_hat[i][0] / se_i
        t_stats.append(t_i)
        
        # p-value = 2 * (1 - CDF(|t|))
        p_i = 2 * (1 - stats.t.cdf(abs(t_i), n - p - 1))
        p_values.append(p_i)
        
        # CI = beta +/- t_crit * SE
        ci_lower = beta_hat[i][0] - t_crit * se_i
        ci_upper = beta_hat[i][0] + t_crit * se_i
        conf_intervals.append((ci_lower, ci_upper))
        
    return {
        "Standard_Errors": se,
        "t_statistics": t_stats,
        "p_values": p_values,
        "CI_95": conf_intervals
    }


# X: Ma tran thiet ke kich thuoc nx(p+1) voi cot dau toan 1
# Tra ve: Danh sach VIF cho tung bien doc lap (bo qua intercept o cot dau)
def vif(X):
    if not is_matrix(X):
        raise ValueError("X phai la ma tran")
        
    n = len(X)
    n_cols = len(X[0])
    
    if n_cols < 2:
        return []
        
    vif_list = []
    for j in range(1, n_cols):
        # Y_j la cot j cua X (kich thuoc nx1)
        y_j = [[row[j]] for row in X]
        # X_j la ma tran con lai cua X (kich thuoc nxp), giu lai cot intercept o index 0
        X_j = [[row[k] for k in range(n_cols) if k != j] for row in X]
        
        if len(X_j[0]) == 1:
            # Chi co 1 bien doc lap duy nhat, nghia la X_j chi chua intercept. VIF = 1.0.
            vif_val = 1.0
        else:
            # Hoi quy y_j theo X_j
            beta_j, _ = ols_fit(X_j, y_j)
            
            # Tinh gia tri du doan
            y_hat_j = mat_mul(X_j, beta_j)
            
            # Tinh R-squared bang cach goi model_metrics.
            # p_j la so dac trung cua X_j (khong tinh intercept), bang len(X_j[0]) - 1
            p_j = len(X_j[0]) - 1
            metrics_j = model_metrics(y_j, y_hat_j, p_j)
            r2_j = metrics_j["R_squared"]
            
            denominator = 1.0 - r2_j
            if denominator <= 1e-12:
                vif_val = float('inf')
            else:
                vif_val = 1.0 / denominator
                
        vif_list.append(vif_val)
        
    return vif_list

