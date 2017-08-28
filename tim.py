from scikits.audiolab import Sndfile
from scikits.talkbox.features import mfcc
import numpy as np

f = Sndfile('./SA1.WAV', 'r')
data = f.read_frames(20000)


print(len(data))

ceps, mspec, spec = mfcc(data,fs=16000)

np.savetxt("./SA1.csv",ceps,delimiter=",")
