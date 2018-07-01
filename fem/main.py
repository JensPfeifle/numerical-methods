import matplotlib.pyplot as plt
import numpy as np

g = np.sin

x = np.linspace(0,3*np.pi,100)

y = g(x)

plt.plot(x,y)
plt.show()