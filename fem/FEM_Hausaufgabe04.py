"""
Based on Example 5 SIAM Workshop October 18, 2014, M. M. Sussman
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from collections import namedtuple

N = 2  # number of elements
le = 0.  # left end
re = 4.  # right end
dx = re / N  # element size
Ndof = 2 * N + 1  # number of degrees of freedom
x = np.linspace(le, re, Ndof)
theta0 = 273
q0 = 100

""" RB """
# Dirichlet (1. Art)
boundaries = []
RB1 = namedtuple('RB1', ['knoten', 'wert'])
RB2 = namedtuple('RB2', ['knoten', 'wert'])
boundaries.append(RB1(Ndof-1, theta0))
boundaries.append(RB2(0, q0))

""" Elementzusammenhangstabelle """
EZHT = []
EZHT.append(('e', 'n1', 'n2', 'n3'))
for e in range(N):
    EZHT.append((e + 1, (2 * e, 2 * e + 1, 2 * e + 2)))

# shape functions
phi = [lambda xi: 2.0 * (xi - 0.5) * (xi - 1.0),
       lambda xi: 4.0 * xi * (1.0 - xi),
       lambda xi: 2.0 * xi * (xi - 0.5)]

# derivative of shape functions w.r.t. xi
dphi = [lambda xi: 2.0 * (xi - 0.5) + 2.0 * (xi - 1.0),
        lambda xi: -4.0 * xi + 4.0 * (1.0 - xi),
        lambda xi: 2.0 * xi + 2.0 * (xi - 0.5)]

# Gauss points and weights
gausspts = np.array(
    (.112701665379258311482073460022, .5, .887298334620741688517926539978))
gausswts = np.array((5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0))

""" Elementsteifigkeiten """
for e in range(1, N+1):
    # compute elemental matrix for \int \phi' \phi'
    Ke = np.zeros([3, 3])
    for i in range(3):
        f = np.array([dphi[0](gausspts[i]), dphi[1](gausspts[i]),
                      dphi[2](gausspts[i])])
        Ke += gausswts[i] / dx * np.outer(f, f)

# Assemble the constant elemental terms of the stiffness matrix A
K = np.zeros([Ndof, Ndof])
f = np.zeros([Ndof, 1])
for e in range(1, N + 1):
    for a in range(0, 3):
        n1 = EZHT[e][1][a]
        f[n1] = f[n1]
        for b in range(0, 3):
            n2 = EZHT[e][1][b]
            K[n1, n2] = K[n1, n2] + Ke[a, b]

print("K=\n", K)

for bc in [e for e in boundaries if type(e) is RB2]:
    f[bc.knoten] = bc.wert
print("f=\n", f)

""" Partitionierung """
LE = np.zeros(Ndof)
for bc in [e for e in boundaries if type(e) is RB1]:
    LE[bc.knoten] = 1
LF = np.zeros([Ndof,])


# assemble RHS vector
rhsvec = np.zeros(Ndof)
#for i in range(N):
#    rhsvec[2 * i: 2 * i + 3] += int_f_phi(i, rhsf)
#print(rhsvec)

""" Solve and plot """
#u = la.solve(K, rhsvec)

#plt.plot(x, u, 'b')
#plt.show()
