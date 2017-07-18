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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import *
from sklearn.svm import SVC


'''
sample_rate, X = scipy.io.wavfile.read("./test.wav")
X = X[4000:8000]
ceps, mspec, spec = mfcc(X,fs=11025)
print(len(X))
print(np.shape(ceps))
print(np.shape(mspec))
print(np.shape(spec))
'''

def coeff(arr):
	ceps, mspec, spec = mfcc(arr,fs=11025)
	print np.shape(ceps)
	auxceps = np.zeros(325)
	for i in range(0,25):
		for j in range(0,13):
			auxceps[13*i+j]=ceps[i][j]
	return(auxceps)


table=eulerlib.numtheory.Divisors()

def process(i,j):
	f = Sndfile('../PDAs/00'+str(i)+'/PDAs0'+str(i)+'_0'+str(j)+'_1.wav', 'r')
	data = f.read_frames(15000)
	data = autocorr(data)
	norm = 0
	for i in range(0,20):
		vec = data[i*500:i*500+4200]
		if (np.linalg.norm(vec)>=norm):
			currvec = vec
			norm = np.linalg.norm(vec)
	return(currvec)

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

frames = [210, 208, 180, 200]
mainvec = []
for i in range(0,4):
	#print(table.divisors(frames[i]))
	mainvec.append(table.divisors(frames[i]))
#print(mainvec)
mainvec = np.concatenate(mainvec)
#print(mainvec)
mainvec = list(set(mainvec))
mainvec.sort()
#print(mainvec)
print(len(mainvec))
matlist = []
for i in range(0,len(mainvec)):
	matlist.append(hlp.matmake(mainvec[i]))
#print(np.shape(matlist[3]))



def maxframeproject(arr):
	features = np.zeros(len(mainvec))
	ham = np.hamming(len(arr))
	auxarr = np.multiply(arr,ham)
	auxarr2 = np.zeros(len(arr))
	for i in range(1,len(arr)):
		auxarr2[i] = auxarr[i] - 0.5*auxarr[i-1]
	auxarr = auxarr2
	for i in range(0,len(mainvec)):
		temp = mainvec[i]
		rem = len(arr)%int(temp)
		rep = len(arr)/int(temp)
		lwin = (rem-1)/2
		uwin = lwin + rep*temp
		aux = subsum(auxarr[lwin:uwin],temp)
		#print(rep)
		features[i] = (float(np.dot(np.transpose(aux),np.matmul(matlist[i],aux)))/float(rep*temp))
	features = features[1:34]
	aux = [np.log(((percentileofscore(features, i, 'weak'))/100)+5e-1) for i in features]
	features = aux
	const = np.mean(features)
	sd = np.std(features)
	m1 = np.amin(features)
	m2 = np.amax(features)
	for i in range(0,len(features)):
		features[i] = 2*((features[i]-m1)/(m2-m1)) - 1
	return(features)
		

def fullframeproj(arr,maxf=210, fullf=660):
	ret = np.zeros(fullf)
	for i in range(0,20):
		aux2 = maxframeproject(arr[i*maxf:(i+1)*maxf])
		for j in range(0,33):
			ret[i*33+j]=aux2[j]
	return(ret)

refvec = np.zeros(84000)
datavec = np.zeros(163200)
datavec = np.reshape(datavec,(4800,34))
refvec = np.reshape(refvec,(6000,14))
for i in range(4,10):
	for j in range(10,50):
		data = process(i,j)
		#plt.plot(data)
		#plt.show()
		feat = fullframeproj(data)
		feat2 = coeff(data)
		for w in range(0,20):
			for z in range(0,33):
				datavec[20*((i-4)*40+(j-10))+w][z] = feat[33*w+z]
			datavec[20*((i-4)*40+(j-10))+w][33] = i	
		for w in range(0,25):
			for z in range(0,13):
				refvec[25*((i-4)*40+(j-10))+w][z] = feat2[13*w+z]
			refvec[25*((i-4)*40+(j-10))+w][13] = i	

'''
for i in range(0,33):
	pts = []
	for z in range(0,240):
		for j in range(0,20):
			pts.append(datavec[z][33*j+i])
	mean = np.mean(pts)
	sd = np.std(pts)
	for z in range(0,240):
		for j in range(0,20):
			datavec[z][33*j+i] = (datavec[z][33*j+i] - mean)/sd
'''
#nsamplebound = 6
classif =  GaussianNB()
classif2 =  GaussianNB()
classif2.fit(datavec[:,:33], datavec[:,33])
classif.fit(refvec[:,:13], refvec[:,13])                    
#classif = OneVsRestClassifier(LinearSVC(random_state=0)).fit(datavec[:,:760], datavec[:,760])


ncorr = 0
nerr = 0
fcorr = 0
ferr = 0
error = np.zeros(240)
error2 = np.zeros(240)


for i in range(4,10):
	for j in range(50,90):
		data = process(i,j)
		#feat = fullframeproj(data)
		feat2 = coeff(data)
		temp = np.zeros(6)
		for w in range(0,25):
			est = classif.predict(feat2[13*w:13*(w+1)])
			if (est==i):
				fcorr = fcorr+1
			else:
				ferr = ferr+1
			temp[int(est-4)] = temp[int(est-4)]+1
		idx = 0
		for w in range(0,6):
			if(temp[w]>=temp[idx]):
				idx = w
		idx = idx+4
		if (idx==i):
			ncorr+=1
		else:
			nerr+=1
		error[(i-4)*40+(j-50)] = float(ncorr)/float(ncorr+nerr)

ref = ncorr
ref2 = nerr

ncorr = 0
nerr = 0
fcorr2 = 0
ferr2 = 0

for i in range(4,10):
	for j in range(50,90):
		data = process(i,j)
		feat = fullframeproj(data)
		#feat2 = coeff(data)
		temp = np.zeros(6)
		for w in range(0,20):
			est = classif2.predict(feat[33*w:33*(w+1)])
			if (est==i):
				fcorr2 = fcorr2+1
			else:
				ferr2 = ferr2+1
			temp[int(est-4)] = temp[int(est-4)]+1
		idx = 0
		for w in range(0,6):
			if(temp[w]>=temp[idx]):
				idx = w
		idx = idx+4
		if (idx==i):
			ncorr+=1
		else:
			nerr+=1
		error2[(i-4)*40+(j-50)] = float(ncorr)/float(ncorr+nerr)

plt.plot(error,'r',error2,'b')
plt.ylabel('Best MFCC vs Ramanujan model')
plt.xlabel('sample ID')
plt.show()
		
print (ncorr)
print (nerr)
print (ref,ref2)
print (fcorr,ferr,fcorr2,ferr2)


