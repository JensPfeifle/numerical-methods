"""
SIAM Workshop October 18, 2014, M. M. Sussman
alexander.py: bad copy example
"""

import copy as cp

x = [1, 2]
y = [3, 4, x]
z = y
print "x=",x," y=",y," z=",z

c=cp.copy(y)
d=cp.deepcopy(y)
print "y=",y," z=",z," c=",c," d=",d

y[0] = '*'
print "y=",y," z=",z," c=",c," d=",d

z[2][0] = 9
print "x=",x," y=",y," z=",z," c=",c," d=",d

c[2][1] = 'c'
print "x=",x," y=",y," z=",z," c=",c," d=",d
