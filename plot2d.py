def plot_2d_unit(ax, scale=1):
    assert (ax.name == 'rectilinear')
    L = scale * 1
    ax.arrow(0, 0, 0, L, color='gray', head_width=0.1, head_length=0.1)
    ax.arrow(0, 0, L, 0, color='gray', head_width=0.1, head_length=0.1)
    return ax


def plot_vector(ax, v, **plotargs):
    assert (ax.name == 'rectilinear')
    ax.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1, **plotargs)
    return ax


def plot_vectors(ax, vectors, **plotargs):
    assert (ax.name == 'rectilinear')
    num_vectors = vectors.shape[1]
    for n in range(num_vectors):
        this_vect = vectors[:, n]
        ax.arrow(0, 0, this_vect[0], this_vect[1], head_width=0.1, head_length=0.1, **plotargs)
    return ax


def plot_solve(ax, solver, A, b, x0, **kwargs):
    """" Solver must take A,b,x0, and maxiter parameters
            and return x, iteration number, and residual eps
    """
    x, k, eps = solver(A, b, x0=x0, maxiter=1)
    dx, dy = x - x0
    x_, y_ = x0
    ax.arrow(x_, y_, dx, dy, head_width=0.1, head_length=0.1, **kwargs)
    # ax.scatter(x_, y_, color='r')
    while eps > 0.3:
        oldx = x.copy()
        x, k, eps = solver(A, b, x0=x, maxiter=1)
        dx, dy = x - oldx
        x_, y_ = oldx
    ax.arrow(x_, y_, dx, dy, head_width=0.1, head_length=0.1, **kwargs)
    finalx, finaly = x
    ax.scatter(finalx, finaly, color='g')
    return ax
