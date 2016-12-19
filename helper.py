import scipy 
import numpy as np
import eulerlib as euler

# test for c_10

pi = 3.14159265359
epsilon = 0.0001

def c_10(n):
	return (np.cos(2*pi*1*n/10)+np.cos(2*pi*3*n/10)+np.cos(2*pi*7*n/10)+np.cos(2*pi*9*n/10))

for i in range(0,10):
	print np.rint(c_10(i))


