import numpy as np

def crout_decomposition(A):
    
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for j in range(n):
        U[j, j] = 1  # Diagonal utama U adalah 1
        for i in range(j, n):
            L[i, j] = A[i, j] - np.dot(L[i, :j], U[:j, j])  # Menghitung elemen L
        for i in range(j+1, n):
            U[j+1:, j] = (A[j+1:, j] - np.dot(L[j+1:, :j], U[:j, j])) / L[j, j]  # Menghitung elemen U
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
    
    L, U = crout_decomposition(A)
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
