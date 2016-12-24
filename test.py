import numpy as np
from scikits.audiolab import Sndfile
import soundfile as sf
import matplotlib.pyplot as plt
import helper as hlp
import math

#testarr = [2,1,2,3,4,5,6,7]

def project(givenarr,q):
  num = int(np.rint(math.pow(2,q)));
  arr = np.zeros(num);
  aux = len(givenarr)/num;
  for i in range(0, aux):
    for j in range(0, num):
      arr[j] = arr[j] + givenarr[i*num+j];
  mat = hlp.makematrix(q)/num;
  #print(mat)
  vec = np.dot(mat,arr);
  #print (arr)
  #print (vec)
  val = np.dot(np.transpose(arr),vec)
  val = (float(num)/float(len(givenarr)))*val;    
  return (val)
  
#print(hlp.makematrix(4)) 

#project(testarr,0)
#project(testarr,1)
#project(testarr,2)
#project(testarr,3) 

data, samplerate = sf.read('./9.wav')
data = data[:1024] # till 2^10
proj = np.zeros(11)
for i in range(0,11):
  proj[i] = project(data,i)
  
#print(np.sum(proj))
#print(np.linalg.norm(data))  

#print(len(data))

plt.plot(proj)
plt.show()


