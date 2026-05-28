# Chu y: Toan bo ham helper duoi day co dau vao va dau ra la 2D list

def mat_mul(A, B):
    if not is_matrix(A) or not is_matrix(B):
        raise ValueError("Khong phai ma tran")

    if len(A[0]) != len(B):
        raise ValueError("Kich thuoc khong phu hop de nhan")
    
    m = len(A)
    p = len(A[0])
    n = len(B[0])
    
    C = [[sum(A[i][k] * B[k][j] for k in range(p)) for j in range(n)] for i in range(m)]
    
    return C

def mat_sub(A, B):
    if not is_matrix(A) or not is_matrix(B):
        raise ValueError("Khong phai ma tran")
    m, n = len(A), len(A[0])
    if m != len(B) and n != len(B[0]):
        raise ValueError("Kich thuoc khong phu hop de tru")
    
    C = [[A[i][j] - B[i][j] for j in range(n)] for i in range(m)]
    
    return C

def mat_inverse(A):
    if not is_matrix(A):
        raise ValueError("Khong phai ma tran")
    
    m, n = len(A), len(A[0])
    if m != n:
        raise ValueError("Khong the nghich dao ma tran")
    
    # Tao ma tran [A | I]
    aug = []
    for i in range(n):
        row = [float(x) for x in A[i]] + [1.0 if i == j else 0.0 for j in range(n)]
        aug.append(row)
        
    # Phep khu Gauss-Jordan
    for i in range(n):
        pivot = aug[i][i]
        
        if pivot == 0:
            for k in range(i + 1, n):
                if aug[k][i] != 0:
                    aug[i], aug[k] = aug[k], aug[i]
                    pivot = aug[i][i]
                    break
            else:
                raise ValueError("Ma tran suy bien, khong the nghich dao")
        
        # Chuan hoa dong chua pivot
        for j in range(2 * n):
            aug[i][j] /= pivot
            
        # Khu cac phan tu con lai trong cot
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2 * n):
                    aug[k][j] -= factor * aug[i][j]
                    
    inv = [[aug[i][j] for j in range(n, 2 * n)] for i in range(n)]
    
    return inv

def mat_trans(A):
    if not is_matrix(A):
        raise ValueError("Khong phai ma tran")

    m, n = len(A), len(A[0])
    A_t = [[A[i][j] for i in range(m)] for j in range(n)]
    
    return A_t

# Tra ve True cho list 2D
def is_matrix(param):
    if not isinstance(param, list) or not param:
        return False
    
    if not isinstance(param[0], list):
        return False
    
    first_row_len = len(param[0])
    return all(isinstance(row, list) and len(row) == first_row_len for row in param)

# Tra ve ma tran don vi I co kich thuoc n x n
def mat_identity(n):
    I = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                I[i][j] = 1
    return I          

def mat_add(A, B):
    if not is_matrix(A) or not is_matrix(B):
        raise ValueError("Khong phai ma tran")
    m, n = len(A), len(A[0])
    if m != len(B) and n != len(B[0]):
        raise ValueError("Kich thuoc khong phu hop de cong")
    
    C = [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]
    
    return C     

def mat_scalar_mul(A, scalar):
    if not is_matrix(A):
        raise ValueError("Khong phai ma tran")
    
    m, n = len(A), len(A[0])
    C = [[A[i][j] * scalar for j in range(n)] for i in range(m)]
    
    return C     

# UNIT TESTS
import unittest

class TestMatrixHelper(unittest.TestCase):
    
    def test_mat_mul(self):
        # Test 1: Ma tran vuong x Ma tran vuong
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        self.assertEqual(mat_mul(A, B), [[19, 22], [43, 50]])
        
        # Test 2: Ma tran chu nhat
        C = [[1, 2, 3], [4, 5, 6]] # 2x3
        D = [[7, 8], [9, 10], [11, 12]] # 3x2
        self.assertEqual(mat_mul(C, D), [[58, 64], [139, 154]])

    def test_mat_inverse(self):
        # Test 1: Ma tran 2x2
        A = [[4, 7], [2, 6]]
        # A^-1 = 1/10 * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
        invA = mat_inverse(A)
        self.assertAlmostEqual(invA[0][0], 0.6)
        self.assertAlmostEqual(invA[1][1], 0.4)
        
        # Test 2: Ma tran 3x3 yeu cau hoan doi dong (pivot = 0)
        B = [[0, 1, 2], [1, 0, 3], [4, -3, 8]]
        invB = mat_inverse(B)
        # Kiem tra bang cach nhan lai voi ma tran goc xem co ra don vi khong
        identity = mat_mul(B, invB)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(identity[i][j], 1.0 if i==j else 0.0)

    def test_mat_add_sub(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        # Add
        self.assertEqual(mat_add(A, B), [[6, 8], [10, 12]])
        # Sub
        self.assertEqual(mat_sub(B, A), [[4, 4], [4, 4]])

    def test_mat_trans(self):
        # Test 1: Vuong
        self.assertEqual(mat_trans([[1, 2], [3, 4]]), [[1, 3], [2, 4]])
        # Test 2: Chu nhat
        self.assertEqual(mat_trans([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])

    def test_is_matrix(self):
        self.assertTrue(is_matrix([[1, 2], [3, 4]]))
        self.assertFalse(is_matrix([1, 2]))
        self.assertFalse(is_matrix([[1, 2], [3]])) # Khong deu

if __name__ == "__main__":
    unittest.main()