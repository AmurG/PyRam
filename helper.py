import scipy 
import numpy as np
import eulerlib as euler
import math

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
    
arr = [2,0,-2,0]

for i in range(0,4):
    circulant(arr)
    print(arr)      
