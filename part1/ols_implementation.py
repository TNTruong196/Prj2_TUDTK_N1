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
    denominator = (rss / (n - p - 1))
    if denominator < 1e-15:
        f_stat = float('inf')
        f_p_val = 0.0
    else:
        f_stat = ((tss - rss) / p) / denominator
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


# Test

import unittest

def _almost_equal_matrix(A, B, tol=1e-6):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False
    for i in range(len(A)):
        for j in range(len(A[0])):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True

class TestOLSImplementation(unittest.TestCase):

    def setUp(self):
        # Dữ liệu giả lập đơn giản
        self.X = [
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
        self.y = [[2.0], [4.1], [5.9], [8.2]]
        self.beta_hat, self.sigma2 = ols_fit(self.X, self.y)

    # ==================================
    # Tests for hat_matrix()
    # ==================================
    def test_hat_matrix_is_symmetric(self):
        """Kiểm tra tính chất đối xứng của ma trận Hat (H^T = H)."""
        H = hat_matrix(self.X)
        H_t = mat_trans(H)
        self.assertTrue(_almost_equal_matrix(H, H_t), "Ma trận Hat phải đối xứng")
        print("TEST PASSED: hat_matrix() đối xứng (H^T = H)")

    def test_hat_matrix_is_idempotent(self):
        """Kiểm tra tính chất lũy đẳng của ma trận Hat (H*H = H)."""
        H = hat_matrix(self.X)
        H_squared = mat_mul(H, H)
        self.assertTrue(_almost_equal_matrix(H, H_squared), "Ma trận Hat phải lũy đẳng")
        print("TEST PASSED: hat_matrix() lũy đẳng (H*H = H)")

    # ==================================
    # Tests for coef_inference()
    # ==================================
    def test_coef_inference_structure(self):
        """Kiểm tra cấu trúc đầu ra của hàm coef_inference."""
        inference = coef_inference(self.X, self.y, self.beta_hat, self.sigma2)
        num_coefs = len(self.beta_hat)
        
        self.assertIn("Standard_Errors", inference)
        self.assertIn("t_statistics", inference)
        self.assertIn("p_values", inference)
        self.assertIn("CI_95", inference)
        
        self.assertEqual(len(inference["Standard_Errors"]), num_coefs)
        self.assertEqual(len(inference["t_statistics"]), num_coefs)
        self.assertEqual(len(inference["p_values"]), num_coefs)
        self.assertEqual(len(inference["CI_95"]), num_coefs)
        print("TEST PASSED: coef_inference() trả về đúng cấu trúc")

    def test_coef_inference_against_statsmodels(self):
        """So sánh kết quả standard errors và t-stats với thư viện statsmodels."""
        import statsmodels.api as sm
        import numpy as np

        # Chuyển sang numpy array để dùng statsmodels
        X_np = np.array(self.X)
        y_np = np.array(self.y)
        
        # Fit mô hình bằng statsmodels
        sm_model = sm.OLS(y_np, X_np).fit()
        
        # Lấy kết quả từ hàm tự cài đặt
        custom_inference = coef_inference(self.X, self.y, self.beta_hat, self.sigma2)
        
        # So sánh Standard Errors
        for i in range(len(self.beta_hat)):
            custom_se = custom_inference["Standard_Errors"][i]
            sm_se = sm_model.bse[i]
            self.assertAlmostEqual(custom_se, sm_se, places=5, msg=f"SE của beta_{i} không khớp")

        # So sánh t-statistics
        for i in range(len(self.beta_hat)):
            custom_t = custom_inference["t_statistics"][i]
            sm_t = sm_model.tvalues[i]
            self.assertAlmostEqual(custom_t, sm_t, places=5, msg=f"t-statistic của beta_{i} không khớp")
            
        print("✅ TEST PASSED: coef_inference() cho kết quả khớp với statsmodels")

class TestVIF(unittest.TestCase):
    
    def test_vif_no_multicollinearity(self):
        """
        Verify VIF is low (close to 1.0) when independent variables are not correlated.
        """
        # Independent variables: x1 and x2 (nearly orthogonal)
        # design matrix X contains intercept column at index 0
        X = [
            [1.0, -2.0, 4.0],
            [1.0, -1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 4.0],
            [1.0, 3.0, 9.0]
        ]
        
        vif_results = vif(X)
        
        # There should be 2 VIF values (one for x1, one for x2)
        self.assertEqual(len(vif_results), 2)
        # All VIFs should be low (< 2.0)
        for val in vif_results:
            self.assertLess(val, 2.0)
            self.assertGreaterEqual(val, 1.0)

    def test_vif_high_multicollinearity(self):
        """
        Verify VIF is high (> 10.0) when independent variables are highly correlated.
        """
        # x2 is x1 + tiny noise, which creates strong multicollinearity
        X = []
        for i in range(1, 15):
            x1 = float(i)
            noise = 0.01 if i % 2 == 0 else -0.01
            x2 = x1 + noise
            X.append([1.0, x1, x2])
            
        vif_results = vif(X)
        
        self.assertEqual(len(vif_results), 2)
        # At least one VIF should be greater than 10.0 (multicollinearity threshold)
        self.assertTrue(any(val > 10.0 for val in vif_results))

    def test_vif_against_sklearn(self):
        from sklearn.linear_model import LinearRegression
        import numpy as np

        """
        Verify the mathematical correctness of our VIF values against scikit-learn LinearRegression.
        """
        X = [
            [1.0, 1.2, 3.4, 2.1],
            [1.0, 2.3, 1.1, 4.5],
            [1.0, 0.5, 4.2, 1.3],
            [1.0, 3.1, 2.2, 5.0],
            [1.0, 4.0, 1.5, 6.2],
            [1.0, 1.8, 3.0, 2.9]
        ]
        
        custom_vifs = vif(X)
        self.assertEqual(len(custom_vifs), 3)
        
        # Calculate reference VIF using sklearn
        X_arr = np.array(X)
        for j in range(1, 4):
            y_j = X_arr[:, j]
            # Select columns other than intercept (0) and the current column j
            cols_j = [k for k in range(X_arr.shape[1]) if k != j]
            X_j = X_arr[:, cols_j]
            
            # Fit regression
            # fit_intercept=False because X_j already contains the intercept column at index 0 (if j != 0)
            model = LinearRegression(fit_intercept=False)
            model.fit(X_j, y_j)
            
            # Compute R-squared
            y_j_hat = model.predict(X_j)
            rss = np.sum((y_j - y_j_hat) ** 2)
            tss = np.sum((y_j - np.mean(y_j)) ** 2)
            r2 = 1.0 - (rss / tss)
            
            ref_vif = 1.0 / (1.0 - r2)
            
            self.assertAlmostEqual(custom_vifs[j-1], ref_vif, places=5)

    def test_vif_perfect_collinearity(self):
        """Kiểm tra VIF khi có đa cộng tuyến hoàn hảo (phải trả về inf)."""
        X = [
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 4.0],
            [1.0, 3.0, 6.0]
        ]
        vif_results = vif(X)
        self.assertEqual(vif_results[1], float('inf'))

class TestModelMetrics(unittest.TestCase):
    def test_metrics_simple_case(self):
        y = [[1.0], [2.0], [3.0]]
        y_hat = [[1.1], [1.9], [3.0]]
        metrics = model_metrics(y, y_hat, 1)
        self.assertGreater(metrics["R_squared"], 0.9)
        self.assertIn("F_statistic", metrics)

    def test_metrics_perfect_fit(self):
        y = [[1.0], [2.0]]
        y_hat = [[1.0], [2.0]]
        # P=1, n=2 => df residuals = 2-1-1 = 0. Warning: this might trigger div by zero in some impls, 
        # but let's check with n=3
        y = [[1.0], [2.0], [3.0]]
        y_hat = [[1.0], [2.0], [3.0]]
        metrics = model_metrics(y, y_hat, 1)
        self.assertEqual(metrics["R_squared"], 1.0)
        self.assertEqual(metrics["RSS"], 0.0)

if __name__ == '__main__':
    print("BẮT ĐẦU CHẠY CÁC BÀI TEST CHO OLS_IMPLEMENTATION VÀ VIF...")
    unittest.main()
