import numpy as np

def householder(x: np.ndarray):
    """ Takes a vector and returns the associated Householder vector
        normalized so that v[0] = 1 """
    N = len(x)
    v = np.concatenate([np.array([1.0]), x[1:]])
    if N == 1:
        sigma = 0
        beta = 0
    else:
        sigma = np.inner(x[1:], x[1:])
        mu = np.sqrt(x[0] ** 2 + sigma)
        # Case to prevent instability from cancellation
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = (-sigma) / (x[0] + mu)  # Formula by Parlett (1971)
        beta = 2 * v[0] ** 2 / (sigma + v[0] ** 2)
        v = v / v[0]
    return v, beta


def qr_decomp(A: np.ndarray):
    """ Takes an MxN Matrix (Numpy array)
        Returns  MxN matrix containing R in the upper right
        and the Householder vectors below the diagonal """
    M, N = A.shape
    for k in range(N):
        v, beta = householder(A[k:M, k])
        Q = np.identity(len(v)) - beta * np.outer(v, v)
        print("Q\n", Q)
        A[k:M, k:N] = np.dot(Q, A[k:M, k:N])
        if k < M:
            A[k + 1:M, k] = v[1:M-k+1]
        print("A\n", A)

    return A

if __name__ == '__main__':
    A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41],[12, -51, 4]], dtype=float)
    qr_decomp(A)
    print(householder(np.array([12,6,-4])))
