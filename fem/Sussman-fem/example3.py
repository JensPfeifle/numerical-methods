"""
Example 3
SIAM Workshop October 18, 2014, M. M. Sussman
example3.py
Gauss integration function
"""

import numpy as np

# Gauss points and weights
gausspts = np.array(\
          (.112701665379258311482073460022,.5,.887298334620741688517926539978))
gausswts = np.array((5.0/18.0,8.0/18.0,5.0/18.0))

def gintegrate(f):
    """
    function for 3-point Gauss integration
    int_0^1( f(x)dx ) = sum_{i=1}^3 w_i f(g_i) 
    NOTE: function of a vector automatically returns vector
    """
    v = np.dot( gausswts,f( gausspts ) )
    return v
    
testfuns=[lambda(x):0.0*x+1]
for i in range(1,7):
    testfuns.append( lambda(x):x**i )

for i in range(7):
    print "integral x**%d = "%i, gintegrate(testfuns[i]), \
          "error = ", abs(gintegrate(testfuns[i]) - 1./(i+1) )