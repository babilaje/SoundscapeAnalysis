# -*- coding: utf-8 -*-
"""
Analyze 10ms samples of a sound clip

@author: Yems
"""

import os
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt
import numpy
import mFeatExtract as mfe
from sklearn.preprocessing import StandardScaler
#from playsound import playsound

# Parameters
f=1 #1-6
FILT=1
fcl = 2000 #Cut-off freq. lower limit
fcu = 20000 #Cut-off freq. upper limit

# Define sound file of interest
#DET 5D0A00B6
#active forest 5D3329A0
#quiet forest 5D34A0A0
path = 'F:\\SS_DATA\\SS_CLAY_0720_beta'
sound = '5D3327C0.WAV'
#for sound in os.listdir(path):   
        
# Load the data and calculate the time of each sample
samplerate, data = wavfile.read(os.path.join(path, sound))
times = numpy.arange(len(data))/float(samplerate)
##data=data[:,1] #mono

if FILT == 1:
    # Filter (working parameters)
    w = fcl / (samplerate/2) #Normalize freq
    b, a = signal.butter(5,w,'high')
    temp = signal.filtfilt(b,a,data)
    w = fcu / (samplerate/2) #Normalize freq
    b, a = signal.butter(5,w,'low')
    data = signal.filtfilt(b,a,temp)
    
X = mfe.FeatArray(data,samplerate,0.1*samplerate, 0.025*samplerate)

#X=StandardScaler().fit_transform(X.T)
plt.plot(X[f-1].T)
print(numpy.mean(X[f-1]))
print(numpy.var(X[f-1]))

#print(sound,numpy.mean(X[f-1]),numpy.var(X[f-1]))