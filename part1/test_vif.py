import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from ols_implementation import vif

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

if __name__ == '__main__':
    unittest.main()
