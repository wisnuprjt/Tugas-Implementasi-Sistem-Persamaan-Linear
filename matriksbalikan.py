import numpy as np

A = np.array([[2, 3, -1],
              [4, -1, 2],
              [1, 2, -3]])

B = np.array([5, 3, 1])

A_inv = np.linalg.inv(A)

X = np.dot(A_inv, B)

print("Solusi x, y, z:")
print(X)
