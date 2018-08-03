""" Lineare Ausgleichsrechnung """

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

# list of tuples (listener_x, listener_y, tan_alpha (est. angle to bird call))
data = [(8, 0, 1), (22, 7, -0.5), (36, 18, 0.5), (10, 20, -1), (13, 10, 0)]

x = [x for x, y, tana in data]
y = [y for x, y, tana in data]

# set up A, b for Ax=b
dims_A = (len(data), 2)
dims_b = (len(data), 1)
A = np.ndarray(dims_A)
b = np.ndarray(dims_b)
for row, values in enumerate(data):
    x, y, tana = values
    A[row, :] = [-1 * tana, 1]
    b[row] = -x * tana + y
    plt.scatter(x, y, label='listener location')

# Normalengleichung A.T A x = A.T b

ATA = A.T.dot(A)
ATb = A.T.dot(b)

bird_x, bird_y = la.solve(ATA, ATb)


# Plotting
pointer_len = 20
for x0, y0, tana in data:
    x_s = x0 - pointer_len / 2 * np.cos(np.arctan(tana))
    x_e = x0 + pointer_len / 2 * np.cos(np.arctan(tana))
    y_s = y0 - pointer_len / 2 * np.sin(np.arctan(tana))
    y_e = y0 + pointer_len / 2 * np.sin(np.arctan(tana))
    plt.plot([x_s, x_e], [y_s, y_e], color='r')
plt.scatter(bird_x, bird_y, label='est. bird location', )
plt.legend()
plt.show()

print('Estimated bird location:')
print('x', x)
print('y', y)
