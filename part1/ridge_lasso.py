from matrix_helper import *
import matplotlib.pyplot as plt
import math 

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


def ridge_trace(X, y, lam=None):
    if lam is None:
        lambdas = [10 ** (-4 + 8 * i / 99) for i in range(100)]
    else:
        lambdas = lam

    if not isinstance(lambdas, list) or len(lambdas) == 0:
        raise ValueError("lam phai la list cac gia tri lambda hoac None")

    for value in lambdas:
        if value <= 0:
            raise ValueError("Tat ca lambda phai > 0 de ve log10(lambda)")

    beta_traces = []

    for value in lambdas:
        beta = ridge_fit(X, y, value)
        beta_traces.append([row[0] if isinstance(row, list) else row for row in beta])

    log_lambdas = [math.log10(value) for value in lambdas]
    num_coef = len(beta_traces[0]) if beta_traces else 0

    for j in range(num_coef):
        coef_values = [beta[j] for beta in beta_traces]
        plt.plot(log_lambdas, coef_values, label=f"beta_{j}")

    plt.title("Ridge Trace")
    plt.xlabel("log10(lambda)")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.grid(True)
    plt.show()

    return beta_traces
    
    