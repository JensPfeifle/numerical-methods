import numpy as np
import doctest


def gauss(A: np.ndarray, b: np.ndarray):
    """ Perform Gaussian elimination with a System Ax=b without pivots
        Returns upper diagonal matrix R and new rhs b

        >>> gauss(A=np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]), \
                 b=np.array([8, -11, -3]))
        (array([[ 2. ,  1. , -1. ],
               [ 0. ,  0.5,  0.5],
               [ 0. ,  0. , -1. ]]), array([8., 1., 1.]))
        """
    M = np.column_stack((A, b))
    M = M.astype(float)
    assert M[0, 0] != 0, "A11 cannot be zero! This function can't pivot.."
    for j in range(M.shape[1] - 1):
        for i in range(j + 1, M.shape[0]):
            assert M[j, j] != 0, "Zero pivot encountered at M{}{}.".format(j + 1, j + 1)
            li = M[i, j] / M[j, j]
            M[i, j:] = M[i, j:] - li * (M[j, j:])
    return M[:, :-1], M[:, -1]


def householder(x: np.ndarray):
    """ Takes a vector and returns the associated Householder vector
        normalized so that v[0] = 1

        >>> householder(np.array([21,5,16]))
        (array([ 1.        , -0.85178039, -2.72569723]), 0.21846092605697376)
        >>> householder(np.array([-13,2,7]))
        (array([ 1.        , -0.07168545, -0.25089908]), 1.8725028717782315)

        """
    x = x.astype(float)
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


def qr(A: np.ndarray):
    """ Takes an MxN Matrix (Numpy array)
        Returns  MxN matrix containing R in the upper right
        and the Householder vectors below the diagonal
        Example from: https://en.wikipedia.org/wiki/QR_decomposition

        >>> qr(A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=float))
        array([[ 14.  ,  21.  , -14.  ],
               [ -3.  , 175.  , -70.  ],
               [  2.  ,  -0.75, -35.  ]])
        >>> qr(A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]]))
        array([[ 14.  ,  21.  , -14.  ],
               [ -3.  , 175.  , -70.  ],
               [  2.  ,  -0.75, -35.  ]])
        """
    A = A.astype(float)
    M, N = A.shape
    for k in range(N):
        v, beta = householder(A[k:M, k])
        Q = np.identity(len(v)) - beta * np.outer(v, v)
        A[k:M, k:N] = np.dot(Q, A[k:M, k:N])
        if k < M:
            A[k + 1:M, k] = v[1:M - k + 1]
    return A


if __name__ == '__main__':
    doctest.testmod()
