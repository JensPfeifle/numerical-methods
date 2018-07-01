# lineare ausgleichsrechnung
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

data = [(8, 0, 1), (22, 7, -0.5), (36, 18, 0.5), (10, 20, -1), (13, 10, 0)]

x = [x for x, y, tana in data]
y = [y for x, y, tana in data]

step = 1
for x0, y0, tana in data:
    x1 = x0 + step * np.cos(np.arctan(tana))
    y1 = y0 + step * np.sin(np.arctan(tana))
    plt.plot([x0, x1], [y0, y1], color='r')

plt.scatter(x, y)
# plt.show()

# set up A, b for Ax=b
dims_A = (len(data), 2)
dims_b = (len(data), 1)
A = np.ndarray(dims_A)
b = np.ndarray(dims_b)
for row, values in enumerate(data):
    x, y, tana = values
    A[row, :] = [-1 * tana, 1]
    b[row] = -x * tana + y

# Normalengleichung A.T A x = A.T b

ATA = A.T.dot(A)
ATb = A.T.dot(b)

x, y = la.solve(ATA, ATb)
plt.scatter(x, y)
plt.show()

print('x', x)
print('y', y)



# Householder Speigelung -> QR Zerlegung
A = np.array([[2, 1, 2, 2], [1, -7, 6, 5], [2, 6, 2, -5], [2, 5, -5, 1]])
#A = np.array([[3, 1], [4, 3]])
x = A[:, 0]
w = np.zeros(A.shape[0])
w[0] = la.norm(x)

v = w - x
v = v[np.newaxis, :].T
vT = v.T
p = v.dot(vT) / vT.dot(v)
H = np.identity(A.shape[0]) - 2 * p
print('R=\n'
      '', H.dot(A))

plt.plot([0, x[0]], [0, x[1]], label='x')
plt.plot([x[0], x[0] + v[0]], [x[1], x[1] + v[1]], label='v')
plt.plot([0, H.dot(x)[0]], [0, H.dot(x)[1]], label='w')
plt.legend()
plt.show()
