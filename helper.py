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
	
def twopower(q,n):
  qeff = math.pow(2,q);
  n = n%qeff;
  if (q==0):
     return 1;
  elif (q==1):
    return math.pow(-1,n);
  elif (n==0):
    return qeff/2;
  elif (n==qeff/2):
    return -qeff/2;
  else:
    return 0;    
      
#test
	
#for i in range(0,5):
#  for j in range(0,int(math.pow(2,i))):
#    print(twopower(i,j))

def circulant(arr):
  if (len(arr)==1):
    return;
  if (len(arr)==2):
    temp = arr[0];
    arr[0] = arr[1];
    arr[1] = temp;
    return;
  else:
    temp = arr[len(arr)-1];
    for i in range(0,len(arr)-1):
      arr[len(arr)-1-i]=arr[len(arr)-2-i];  
    arr[0] = temp;
    return;
    
#arr = [2,0,-2,0]

#for i in range(0,4):
#    circulant(arr)
#    print(arr)
    
def makematrix(q):
  num = int(np.rint(math.pow(2,q)));
  #print(num);
  basecol = np.zeros(num);
  copycol = np.zeros(num);
  for i in range(0, num):
    basecol[i] = twopower(q,i);
    copycol[i] = basecol[i];
  if (q==0):
    return basecol;
  for j in range(1, num):
    circulant(copycol);
    #print(basecol);
    #print(copycol);
    basecol = np.append(basecol,copycol);
    #print (basecol);
  #print (len(basecol))
  #print (np.shape(basecol))  
  basecol = np.reshape(basecol,(num,num));
  return basecol;

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

def matmake(q):
	mat = np.zeros(int(np.rint(math.pow(q,2))))
	mat = np.reshape(mat,(q,q))
	aux = np.zeros(q)
	aux = seqmake(q)
	for i in range(0,q):
		for j in range(0,q):
			mat[i][j] = aux[(i-j+q)%q]
	return(mat)
	
print(matmake(9))	
  
#print(makematrix(0))
