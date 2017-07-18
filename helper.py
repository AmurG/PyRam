import scipy 
import numpy as np
import math
from eulerlib import Divisors
import eulerlib 

# test for c_10

pi = 3.14159265359
epsilon = 0.0001

def c_10(n):
	return (np.cos(2*pi*1*n/10)+np.cos(2*pi*3*n/10)+np.cos(2*pi*7*n/10)+np.cos(2*pi*9*n/10))

#for i in range(0,10):
#	print np.rint(c_10(i))

#Construct the base ramanujan sequence
	
def seqmake(q):
	seq = np.zeros(q)
	for i in range(0,q):
		if (eulerlib.numtheory.gcd(i+1,q)==1):
			for j in range(0,q):
				seq[j]=seq[j]+np.cos(2*pi*(i+1)*j/q)
	for j in range(0,q):
		seq[j] = int(np.rint(seq[j]))
	return(seq)

#print(seqmake(1))

#Construct the matrix

def matmake(q):
	mat = np.zeros(int(np.rint(math.pow(q,2))))
	mat = np.reshape(mat,(q,q))
	aux = np.zeros(q)
	aux = seqmake(q)
	for i in range(0,q):
		for j in range(0,q):
			mat[i][j] = aux[(i-j+q)%q]
	return(mat)
	
#print(matmake(9))	
  
#print(makematrix(0))
