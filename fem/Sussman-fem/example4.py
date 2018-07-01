"""
Example 4
SIAM Workshop October 18, 2014, M. M. Sussman
example4.py
Plotting shape functions
"""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# shape functions
phi = [lambda xi: 2.0 * (xi - 0.5) * (xi - 1.0), \
       lambda xi: 4.0 * xi * (1.0 - xi), \
       lambda xi: 2.0 * xi * (xi - 0.5)]

# derivative of shape functions w.r.t. xi
dphi = [lambda xi: 2.0 * (xi - 0.5) + 2.0 * (xi - 1.0), \
        lambda xi: -4.0 * xi + 4.0 * (1.0 - xi), \
        lambda xi: 2.0 * xi + 2.0 * (xi - 0.5)]

L = 4
N = 2
dx = float(L) / float(N)
Ndof = 2 * N + 1
Nplot = 100
x = np.linspace(0, L, Ndof)
xiplot = np.linspace(0, 1, Nplot)


def plotphi(k, n, hold=True):
    """
    plot one of the shape functions
    """
    xplot = (k + xiplot) * dx
    plt.plot(xplot, phi[n](xiplot), label="elt%d num%d" % (k, n))

def plotdphi(k, n):
    """
    plot one of the shape function derivatives
    """
    xplot = (k + xiplot) * dx
    plt.plot(xplot, dphi[n](xiplot), label="elt%d num%d" % (k, n))


plotphi(2, 0)
plotphi(2, 1)
plotphi(2, 2)
plotdphi(1, 0)
plotdphi(1, 1)
plotdphi(1, 2)
plt.legend(loc='upper center')

plt.show()
