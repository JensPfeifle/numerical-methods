"""
Example 2
SIAM Workshop October 18, 2014, M. M. Sussman
example2.py
functions, flow control and import
"""

import numpy as np

def sine(x):
    """
    compute sin(x) to error of 1.e-10
    using Maclaurin (Taylor) series
    """
    tol=1.e-10
    term=x
    partialSum=term
    n=1
    while abs(term) > tol:  # abs is built-in
        n += 2
        term=(-term) *x*x / (n*(n-1) )
        partialSum += term
        assert( n<10000 ) # not converging!  something is wrong!
    return partialSum

for i in range(10):
    x = i/10.0  # force division to be float
    y = sine(x)
    err = abs(y-np.sin(x))
    if err > 1.e-8:
        print "Something is very wrong!"
    else:
        print "x =", x, " sin(x) =", y, "error =", err
