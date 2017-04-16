import numpy as np
from scikits.audiolab import Sndfile
import soundfile as sf
import matplotlib.pyplot as plt
import helper as hlp
import math
import eulerlib
from time import time
from numpy.fft import fft, ifft
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from scipy import stats
from sklearn.neural_network import MLPClassifier

table=eulerlib.numtheory.Divisors()

def process(i,j):
	f = Sndfile('../PDAs/00'+str(i)+'/PDAs0'+str(i)+'_0'+str(j)+'_1.wav', 'r')
	data = f.read_frames(15000)
	data = data[5000:13192] # 2^n
	data = autocorr(data/np.linalg.norm(data))
	return(data/np.linalg.norm(data))

def timit(i,j):
	f = Sndfile(str(i)+'_'+str(j)+'.wav', 'r')
	data = f.read_frames(22000)
	data = data[5000:21384] # 2^n
	data = autocorr(data/np.linalg.norm(data))
	return(data/np.linalg.norm(data))

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

def subsum(arr,div):
	aux = np.zeros(div)
	for i in range(0,len(arr)/div):
		for j in range(0,div):
			aux[j] = aux[j] + arr[i*div+j]
	return(aux)

#print(subsum(hlp.seqmake(10),1))
	

def genproj(arr):
	div = table.divisors(len(arr))
	en = np.zeros(len(div))
	for i in range(0,len(div)):
		print(div[i])
		aux = subsum(arr,div[i])
		mat = hlp.matmake(div[i])
		en[i] = float(np.dot(np.transpose(aux),np.matmul(mat,aux)))/float(len(arr))
	return(en)

#tst = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
#print(genproj(tst))

'''
timitvec = np.zeros(40)
timitvec = np.reshape(timitvec,(20,2))
temp = np.zeros(15)

for i in range(0,10):
	for j in range(0,2):
		data = timit(i,j)
		for z in range(0,15):
			temp[z] = project(data,z)
		timitvec[2*i+j][0] = temp[14]
		timitvec[2*i+j][1] = temp[13]

print(timitvec)

plt.plot(timitvec[10:12,0],timitvec[10:12,1],'ko', timitvec[12:14,0],timitvec[12:14,1],'wo', timitvec[14:16,0],timitvec[14:16,1],'go', timitvec[16:18,0],timitvec[16:18,1],'ro', timitvec[18:20,0],timitvec[18:20,1],'co')
plt.show()
'''

'''  
datavec = np.zeros(6300)
datavec = np.reshape(datavec,(420,15))
for i in range(4,10):
	for j in range(10,80):
		data = process(i,j)
		#plt.plot(data)
		#plt.show()
			
		for z in range(0,14):
	  		datavec[(i-4)*70+(j-10)][z] = project(data,z)
		datavec[(i-4)*70+(j-10)][14] = i

nsamplebound = 6
classif = OneVsRestClassifier(LinearSVC(random_state=0)).fit(datavec[:,nsamplebound:14], datavec[:,14])

temp = np.zeros(14)
ncorr = 0
nerr = 0

for i in range(4,10):
	for j in range(80,90):
		data = process(i,j)
		for z in range(0,14):
			temp[z] = project(data,z)
		est = classif.predict(temp[nsamplebound:14])
		print(est)
		if (est[0]==i):
			ncorr+=1
		else:
			nerr+=1

print (ncorr)
print (nerr)
'''
