import numpy as np
from scikits.audiolab import Sndfile
from glob import glob
import soundfile as sf
import matplotlib.pyplot as plt
import helper as hlp
import math
import eulerlib
from time import time
from numpy.fft import fft, ifft
from scipy import stats
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
from scipy.stats import *
from sklearn.mixture import GaussianMixture as gmm

table=eulerlib.numtheory.Divisors() #Just making table, goes till 1k, big enough for almost all cases

#Sampling rate is 11025. You want 25 ms segment -> 11025/40 => 275 data points at least. Consider a few possible candidates nearby : 2^5 x 3^2 = 288, (18 divisors) 5*5*5*2*2 = 250,  7*7*3*2 = 294. The maximum of those is 294. So, let us restrict the frame size to 294 for generating up a feature map. We'll now search for 294-sized frames.  We now add 11*13*2 = 286, and 5*17*3 = 255 to nearly fully cover the divisors till 17 ( as the smaller periods contain significant energy post-highpass )

def coeff(arr):
	ceps, mspec, spec = mfcc(arr,fs=11025)
	#print np.shape(ceps)
	auxceps = np.zeros(130)
	for i in range(0,10):
		for j in range(0,13):
			auxceps[13*i+j]=ceps[i][j]
	return(auxceps)

#A helper function that should take as follows : gmix is a list of Scipy GMM objects. For each object in the list the function calculates the score from the GMM Model. The index is the speaker index i.e. the GMM models occur in the speaker order. The highest one is arg-returned.

#The feats arg = dimensionality of a feature vec. The nfeats arg denotes how many feature vecs were fed in for this sample.


def returnscore(gmixlist, featmatrix):
	winidx = 0
	setscore = -1e13
	for i in range(0,len(gmixlist)):
		chosen = gmixlist[i]
		score = chosen.score(featmatrix)
		if (score>setscore):
			winidx = i
			setscore = score
	return winidx

#Wrapper to load files from the PDAs dataset, k = number of speech frames deemed suitable

def ksel(filename,k=6,read=14700):
	f = Sndfile(filename, 'r')
	data = f.read_frames(read)
	data = autocorr(data)
	#print(data)
	norm = 0
	eng = np.zeros(60)
	eng = np.reshape(eng,(30,2))	
	for i in range(0,30):
		vec = data[i*294:(i+1)*294]
		eng[i][0] = np.linalg.norm(vec)
		eng[i][1] = i
	#print(eng[:,1])
	eng = eng[eng[:,0].argsort()]
	#print(eng[:,1])
	returnarr = np.zeros(k*294)
	#returnarr = np.reshape(returnarr,(k,294))
	for i in range(0,k):
		idx = int(eng[i][1])	
		for j in range(0,294):
			returnarr[i*294+j] = data[idx*294+j]
	return(returnarr)

def process(i,j,k=6,readin=14700):
	filename = str('../PDAs/00'+str(i)+'/PDAs0'+str(i)+'_0'+str(j)+'_1.wav')
	return ksel(filename,k,readin)

#Call to get the TIMIT speakers as an array, all 630

def timit(k=6,readin=14000):
	dirs = glob("../TIMIT/*/")
	timitarr = np.zeros((6300,294*k))
	for i in range(0,630):
		wavs = glob(str(dirs[i])+"*.WAV")
		print(dirs[i])
		for j in range(0,10):
			address = wavs[j]
			get = ksel(address,read=readin)
			for w in range(0,294*k):
				timitarr[10*i+j][w] = get[w]
	return timitarr

def autocorr(x):
    return ifft(fft(x) * fft(x).conj()).real

def subsum(arr,div):
	aux = np.zeros(div)
	for i in range(0,len(arr)/div):
		for j in range(0,div):
			aux[j] = aux[j] + arr[i*div+j]
	return(aux)

def genproj(arr):
	div = table.divisors(len(arr))
	en = np.zeros(len(div))
	for i in range(0,len(div)):
		print(div[i])
		aux = subsum(arr,div[i])
		mat = hlp.matmake(div[i])
		en[i] = float(np.dot(np.transpose(aux),np.matmul(mat,aux)))/float(len(arr))
	return(en)

frames = [288, 250, 294, 286, 255]
mainvec = []
for i in range(0,len(frames)):
	#print(table.divisors(frames[i]))
	mainvec.append(table.divisors(frames[i]))
mainvec = np.concatenate(mainvec)
mainvec = list(set(mainvec))
mainvec.sort()

matlist = []
for i in range(0,len(mainvec)):
	matlist.append(hlp.matmake(mainvec[i]))

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
	features = features[1:]
	aux = [np.log(((percentileofscore(features, i, 'weak'))/100)+5e-1) for i in features]
	features = aux
	const = np.mean(features)
	sd = np.std(features)
	m1 = np.amin(features)
	m2 = np.amax(features)
	for i in range(0,len(features)):
		features[i] = 2*((features[i]-m1)/(m2-m1)) - 1
	return(features)
		

#assume fsize divides arr, i.e. you pass a total array that is made up exactly of fsize-sized frames

def fullframeproj(arr,fsize):
	frames = len(arr)/fsize
	temp = len(mainvec)-1
	ret = np.zeros((frames*temp))
	for i in range(0,frames):
		aux2 = maxframeproject(arr[i*fsize:(i+1)*fsize])
		for j in range(0,temp):
			ret[i*temp+j]=aux2[j]
	return(ret)

#placeholder = timit()
#print(np.shape(placeholder))

print(len(mainvec))

length = len(mainvec)-1

#Uncomment for mini-frame level recogn stats on PDAs

datavec = np.zeros((1440,len(mainvec)))
refvec = np.zeros((2400,14))
for i in range(4,10):
	for j in range(10,50):
		data = process(i,j)
		#plt.plot(data)
		#plt.show()
		feat = fullframeproj(data,294)
		#print(feat)
		feat2 = coeff(data)
		for w in range(0,6):
			for z in range(0,length):
				datavec[6*((i-4)*40+(j-10))+w][z] = feat[length*w+z]
			datavec[6*((i-4)*40+(j-10))+w][len(mainvec)-1] = i	
		for w in range(0,10):
			for z in range(0,13):
				refvec[10*((i-4)*40+(j-10))+w][z] = feat2[13*w+z]
			refvec[10*((i-4)*40+(j-10))+w][13] = i	

ramanclass = []
mfccclass = []
for i in range(0,6):
	clf = gmm(n_components=3, covariance_type = 'diag')
	clf.fit(datavec[240*i:240*(i+1),:length])
	ramanclass.append(clf)
	clf = gmm(n_components=3, covariance_type = 'diag')
	clf.fit(refvec[240*i:240*(i+1),:13])
	mfccclass.append(clf)

ncorr,nerr = 0,0
nsc,nse = 0,0

for i in range(4,10):
	for j in range(50,90):
		data = process(i,j)
		feat = fullframeproj(data,294)
		feat2 = coeff(data)
		for w in range(0,6):
			pred = returnscore(ramanclass,feat[length*w:(w+1)*length])
			if(pred==(i-4)):
				ncorr = ncorr+1
			else:
				nerr = nerr+1
		for w in range(0,10):
			pred = returnscore(mfccclass,feat2[13*w:13*(w+1)])
			if(pred==(i-4)):
				nsc = nsc+1
			else:
				nse = nse+1

	
print(ncorr,nerr,nsc,nse)

'''
timitdata = timit()
datavec = np.zeros((18900,len(mainvec)))
refvec = np.zeros((31500,14))

for i in range(0,630):
	for j in range(0,5):
		data = timitdata[10*i+j,:]
		#plt.plot(data)
		#plt.show()
		feat = fullframeproj(data,294)
		#print(feat)
		feat2 = coeff(data)
		for w in range(0,6):
			for z in range(0,length):
				datavec[6*(i*5+j)+w][z] = feat[length*w+z]
			datavec[6*(i*5+j)+w][len(mainvec)-1] = i	
		for w in range(0,10):
			for z in range(0,13):
				refvec[10*(i*5+j)+w][z] = feat2[13*w+z]
			refvec[10*(i*5+j)+w][13] = i	

ramanclass = []
mfccclass = []
for i in range(0,630):
	clf = gmm(n_components=3, covariance_type = 'full')
	clf.fit(datavec[30*i:30*(i+1),:length])
	ramanclass.append(clf)
	clf = gmm(n_components=3, covariance_type = 'full')
	clf.fit(refvec[50*i:50*(i+1),:13])
	mfccclass.append(clf)

ncorr,nerr = 0,0
nsc,nse = 0,0

for i in range(0,630):
	for j in range(5,10):
		data = timitdata[10*i+j,:]
		feat = fullframeproj(data,294)
		feat2 = coeff(data)
		for w in range(0,6):
			pred = returnscore(ramanclass,feat[length*w:(w+1)*length])
			if(pred==i):
				ncorr = ncorr+1
			else:
				nerr = nerr+1
		for w in range(0,10):
			pred = returnscore(mfccclass,feat2[13*w:13*(w+1)])
			if(pred==(i)):
				nsc = nsc+1
			else:
				nse = nse+1

	
print(ncorr,nerr,nsc,nse)
'''
