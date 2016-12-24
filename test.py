import numpy as np
from scikits.audiolab import Sndfile
import soundfile as sf
import matplotlib.pyplot as plt

data, samplerate = sf.read('./test3.wav')
print (len(data))
print (np.shape(data))
print (data[1:100])
print (samplerate)
plt.plot(abs(np.fft.fft(data[1:30000])))
plt.show()

f = Sndfile('./test3.wav', 'r')


# Sndfile instances can be queried for the audio file meta-data
fs = f.samplerate
nc = f.channels
enc = f.encoding
print(fs)
print(nc)
print(enc)

# Reading is straightfoward
data = f.read_frames(30000)

plt.plot(abs(np.fft.fft(data[1:30000])))
plt.show()

