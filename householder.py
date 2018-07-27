import numpy as np
import scipy.linalg as la

# Householder Speigelung -> QR Zerlegung
def householder_matrix(A):
    x = A[:, 0]
    w = np.zeros(A.shape[0])
    w[0] = la.norm(x)

    v = w - x
    v = v[np.newaxis, :].T
    p = np.dot(v, v.T) / np.dot(v.T, v)
    H = np.identity(A.shape[0]) - 2 * p
    return H


if __name__ == '__main__':
    # my_A = np.array([[2, 1, 2, 2], [1, -7, 6, 5], [2, 6, 2, -5], [2, 5, -5, 1]])
    # Example from https://en.wikipedia.org/wiki/Householder_transformation
    my_A = np.array([[4, 1, -2, 2], [1, 2, 0, 1], [-2, 0, 3, -2], [2, 1, -2, -1]], dtype=float)
    Q1 = householder_matrix(my_A)
    print(Q1)
