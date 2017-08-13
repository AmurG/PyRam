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

table=eulerlib.numtheory.Divisors() #Just making table, goes till 1k, big enough for almost all cases

#Sampling rate is 11025. You want 25 ms segment -> 11025/40 => 275 data points at least. Consider a few possible candidates nearby : 2^5 x 3^2 = 288, (18 divisors) 5*5*5*2*2 = 250,  7*7*3*2 = 294. The maximum of those is 294. So, let us restrict the frame size to 294 for generating up a feature map. We'll now search for 294-sized frames.  We now add 11*13*2 = 286, and 5*17*3 = 255 to nearly fully cover the divisors till 17 ( as the smaller periods contain significant energy post-highpass )

def coeff(arr):
	ceps, mspec, spec = mfcc(arr,fs=11025)
	#print np.shape(ceps)
	auxceps = np.zeros(325)
	for i in range(0,25):
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

def process(i,j,k=6):
	f = Sndfile('../PDAs/00'+str(i)+'/PDAs0'+str(i)+'_0'+str(j)+'_1.wav', 'r')
	data = f.read_frames(14700)
	data = autocorr(data)
	norm = 0
	eng = np.zeros(60)
	eng = np.reshape(eng,(30,2))	
	for i in range(0,30):
		vec = data[i*294:(i+1)*294]
		eng[i][0] = np.linalg.norm(vec)
		eng[i][1] = i
	eng = eng[eng[:,0].argsort()]
	returnarr = np.zeros(k*294)
	returnarr = np.reshape(returnarr,(k,294))
	for i in range(0,k):
		idx = eng[i][1]	
		for j in range(0,294):
			returnarr[i][j] = data[idx*294+j]
	return(returnarr)

#Wrapper to load files from timit, not currently used, commented out
'''
def timit(i,j):
	f = Sndfile(str(i)+'_'+str(j)+'.wav', 'r')
	data = f.read_frames(22000)
	data = data[5000:21384] # 2^n
	data = autocorr(data/np.linalg.norm(data))
	return(data/np.linalg.norm(data))
'''

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
	frames = arr/fsize
	temp = len(mainvec)-1
	ret = np.zeros(frames*temp)
	for i in range(0,frames):
		aux2 = maxframeproject(arr[i*fsize:(i+1)*fsize])
		for j in range(0,temp):
			ret[i*temp+j]=aux2[j]
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


	
