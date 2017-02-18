import numpy as np
from scikits.audiolab import Sndfile
import soundfile as sf
import matplotlib.pyplot as plt
import helper as hlp
import math
from time import time
from numpy.fft import fft, ifft
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from scipy import stats
from sklearn.neural_network import MLPClassifier

#testarr = [2,1,2,3,4,5,6,7]

def autocorr(x):
    return ifft(fft(x) * fft(x).conj()).real

def project(givenarr,q):
  num = int(np.rint(math.pow(2,q)));
  arr = np.zeros(num);
  aux = len(givenarr)/num;
  t1 = time()
  for i in range(0, aux):
    for j in range(0, num):
      arr[j] = arr[j] + givenarr[i*num+j];
  t2 = time()    
  #mat = hlp.makematrix(q)/num;
  t3 = time()
  #print(mat)
  #vec = np.dot(mat,arr);
  vec = np.zeros(len(arr))
  if (q==0):
    vec = arr;
  elif (q==1):
    vec[0] = 0.5*arr[0] - 0.5*arr[1]
    vec[1] = -0.5*arr[0] + 0.5*arr[1]
  else:
    for i in range(0,num/2):
      vec[i] = 0.5*arr[i] - 0.5*arr[num/2+i]
      vec[i+num/2] = -vec[i];    
  t4 = time()
  #print (arr)
  #print (vec)
  val = np.dot(np.transpose(arr),vec)
  t5 = time()
  val = (float(num)/float(len(givenarr)))*val;
  print(t2-t1,t3-t2,t4-t3,t5-t4)    
  return (val)
  
#print(hlp.makematrix(4)) 

#project(testarr,0)
#project(testarr,1)
#project(testarr,2)
#project(testarr,3) 

datavec = np.zeros(6300)
datavec = np.reshape(datavec,(420,15))
for i in range(4,10):
	for j in range(10,80):
		f = Sndfile('../PDAs/00'+str(i)+'/PDAs0'+str(i)+'_0'+str(j)+'_1.wav', 'r')
		data = f.read_frames(15000)
		data = data[5000:13192] # 2^n
		data = data/np.linalg.norm(data)
		data = autocorr(data)
		data = data/np.linalg.norm(data)
		#plt.plot(data)
		#plt.show()
			
		for z in range(0,14):
	  		datavec[(i-4)*70+(j-10)][z] = project(data,z)
		datavec[(i-4)*70+(j-10)][14] = i

#print(datavec[233,:])

#plt.plot(datavec[:70,13],datavec[:70,12],'bo', datavec[70:140,13],datavec[70:140,12],'ro', datavec[140:210,13],datavec[140:210,12],'go', datavec[210:280,13],datavec[210:280,12],'yo', datavec[280:350,13],datavec[280:350,12],'co', datavec[350:420,13],datavec[350:420,12],'ko')
#plt.show()

classif = OneVsRestClassifier(LinearSVC(random_state=0)).fit(datavec[:,9:14], datavec[:,14])

temp = np.zeros(14)
ncorr = 0
nerr = 0

for i in range(4,10):
	for j in range(80,90):
		f = Sndfile('../PDAs/00'+str(i)+'/PDAs0'+str(i)+'_0'+str(j)+'_1.wav', 'r')
		data = f.read_frames(15000)
		data = data[5000:13192] 
		data = data/np.linalg.norm(data)
		data = autocorr(data)
		data = data/np.linalg.norm(data)
		for z in range(0,14):
			temp[z] = project(data,z)
		est = classif.predict(temp[9:14])
		print(est)
		if (est[0]==i):
			ncorr+=1
		else:
			nerr+=1

print (ncorr)
print (nerr)
