"""
Example 1
SIAM Workshop October 18, 2014, M. M. Sussman
example1.py
Python basics
"""

print "Hello"

# integer (4 bytes)
x = -5

# float  (8 bytes)
y0 = 3.14159
y1 = 2.71828

# complex numbers
z = 1.5 + 0.5j
print "real part z = ", z.real
print "imaginary part z = ", z.imag
w = 3 + 4j

ok = True

print "y0 + y1 =", y0 + y1
print "y0 * y1 =", y0 * y1
print "y0 / y1 =", y0 / y1
print "x ** 2 =", x**2

print "(y1 > y0) =", (y1 > y0)

# formatted print, tuple
print "(%f == %f) = %s"%(y0, y1, y0==y1)

# long integer
i = 123456789012345678901234567890L
print "i**2 = ", i**2
