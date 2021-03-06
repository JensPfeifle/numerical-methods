import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la



def jacobi_method(A, b, x0=None, eps=0.1, maxiter=10000):
    assert (A.shape[1] == len(b))
    ndims = len(b)
    if x0 is None:
        x = np.zeros(ndims, dtype=np.float64)
    else:
        assert (len(x0) == len(b))
        x = x0.astype(np.float64)
    b = b.astype(np.float64)
    A = A.astype(np.float64)
    for k in range(0, maxiter):
        r = la.norm(A.dot(x) - b, ord=2)
        if r < eps:
            break
        else:
            xk = x.copy()
            k = k + 1
            for m in range(ndims):
                summe = 0
                for n in range(0, ndims):
                    if n != m:
                        summe += A[m, n] * xk[n]
                x[m] = (b[m] - summe) / A[m, m]
    return x, k, r


def gauss_seidel_method(A, b, x0=None, eps=0.1, maxiter=10000):
    assert (A.shape[1] == len(b))
    ndims = len(b)
    if x0 is None:
        x = np.zeros(ndims, dtype=np.float64)
    else:
        assert (len(x0) == len(b))
        x = x0.astype(np.float64)
    b = b.astype(np.float64)
    A = A.astype(np.float64)
    for k in range(0, maxiter):
        r = la.norm(A.dot(x) - b, ord=2)
        if r < eps:
            break
        else:
            xk = x.copy()
            k = k + 1
            for m in range(ndims):
                summe = 0
                for n in range(0, m):
                    summe += A[m, n] * x[n]
                for n in range(m + 1, ndims):
                    if n != m:
                        summe += A[m, n] * xk[n]
                x[m] = (b[m] - summe) / A[m, m]

    return x, k, r


def GMRES_method(A, b, B, x0=None, eps=0.1, maxiter=100):
    assert (A.shape[1] == len(b))
    ndims = len(b)
    if x0 is None:
        x = np.zeros(ndims, dtype=np.float64)
    else:
        assert (len(x0) == len(b))
        x = x0.astype(np.float64)
    b = b.astype(np.float64)
    A = A.astype(np.float64)
    h = np.array((maxiter, maxiter))
    v = np.array(ndims)
    r = b - A.dot(x)
    print('r0', r)
    z = B.dot(r)
    print('z', z)
    h = la.norm(z, ord=2)
    print('h', h)
    v = z / h
    print('v', v)

    k = 1
    w = B.dot(A).dot(v)
    print('w', w)
    print('vTw', np.dot(np.inner(v, w), v))
    z = w - np.dot(np.inner(v, w), v)
    print('z', z)
    h = la.norm(z, ord=2)
    print('h', h)
    v = np.array([[v], [z / h]])
    print('v', v)

    return v

if __name__ == '__main__':
    A = np.array([[2, 1], [-1, 2]])
    b = np.array([4, -7])
    x0 = np.array([2, 2])
    x, k, eps = jacobi_method(A, b, x0=x0)
    # print('x =', x)
    # print('k =', k)
    # print('eps =', eps)

    x, k, eps = gauss_seidel_method(A, b, x0=x0)
    # print('x =', x)
    # print('k =', k)
    # print('eps =', eps)

    fig = plt.figure(figsize=(8, 8), dpi=200)
    ax = fig.add_subplot(111)
    # axes setup
    limits = 8
    xlim = ylim = [-limits, limits]
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(xlim[0], xlim[1], 1)
    minor_ticks = np.arange(xlim[0], xlim[1], 0.5)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(which='both')

    # plots
    ax.arrow(0, 0, b[0], b[1], head_width=0.1, head_length=0.1, color='green')
    ax.arrow(0, 0, x[0], x[1], head_width=0.1, head_length=0.1, color='green')
    ax = plot_2d_unit(ax, scale=1)
    ax = plot_2d_vectors(ax, A, color='blue')
    ax = plot_solve(ax, jacobi_method, A, b, x0, color='cyan')
    ax = plot_solve(ax, gauss_seidel_method, A, b, x0, color='red')
    plt.show()
    B = np.eye(2)
    v = GMRES_method(A, b, B, x0=x0)
