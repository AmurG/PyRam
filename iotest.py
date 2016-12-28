import numpy as np
from scikits.audiolab import Sndfile
import soundfile as sf
import matplotlib.pyplot as plt
import helper as hlp
import math
from time import time

#data, samplerate = sf.read('./an4.raw')
f = Sndfile('./test.wav', 'r')
print (f.samplerate)
print (f.channels)
data = f.read_frames(16000)
print(data)

data = data/np.linalg.norm(data)
plt.plot(data)
plt.show()



