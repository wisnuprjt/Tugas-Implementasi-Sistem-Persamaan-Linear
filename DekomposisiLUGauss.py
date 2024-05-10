import numpy as np

def lu_decomposition(A):
    
    n = len(A)
    L = np.eye(n)
    U = A.astype(float)  # Mengonversi tipe data menjadi float64
    for k in range(n-1):
        for i in range(k+1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    return L, U

def forward_substitution(L, B):
    
    n = len(L)
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = (B[i] - np.dot(L[i, :i], Y[:i])) / L[i, i]
    return Y

def backward_substitution(U, Y):
    
    n = len(U)
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        X[i] = (Y[i] - np.dot(U[i, i+1:], X[i+1:])) / U[i, i]
    return X

def solve_linear_system(A, B):
    
    L, U = lu_decomposition(A)
    Y = forward_substitution(L, B)
    X = backward_substitution(U, Y)
    return X

# Matriks koefisien
A = np.array([[2, 3, -1],
              [4, -1, 2],
              [1, 2, -3]])

# Matriks hasil
B = np.array([5, 3, 1])

# Menyelesaikan sistem persamaan linier
X = solve_linear_system(A, B)
print("Solusi x, y, z:")
print(X)
