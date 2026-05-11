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