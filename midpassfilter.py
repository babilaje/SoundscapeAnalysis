# -*- coding: utf-8 -*-
"""
Perform MidPass Filter on a wav file to isolate desired freq range
"""
#TEMP - postfilter.wav is at half speed to bring high freq into human range

import os
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
#from playsound import playsound

# Parameters
fcl = 14000 #Cut-off freq. lower limit
fcu =  20000 #Cut-off freq. upper limit

# Define sound file of interest
#sound = 'forest.wav'
#sound = 'buck_snort_wheeze.wav' #5000-15000
#path = 'C:\\Users\\Yems\\Desktop\\repos\\SS_sheperd'
path = 'F:\\SS_DATA\\SS_BBF_beta'
sound = '5D1F9E5D.WAV'

# Load the data and calculate the time of each sample
samplerate, data = wavfile.read(os.path.join(path, sound))
times = np.arange(len(data))/float(samplerate)
##data=data[:,1] #mono

# Filter (working parameters
w = fcl / (samplerate/2) #Normalize freq
b, a = signal.butter(5,w,'high')
data2 = signal.filtfilt(b,a,data)
#w = fcu / (samplerate/2) #Normalize freq
#b, a = signal.butter(5,w,'low')
#data2 = signal.filtfilt(b,a,temp)

# Write WAV files
data2 = np.asarray(data2, dtype=np.int16) #change datatype to int16
wavfile.write('prefilter.WAV',samplerate,data)
wavfile.write('postfilter.WAV',int(samplerate/2),data2) #TEMP

# Export Plot (prefilter)
fig, (og, sg) = plt.subplots(nrows=2)
# Oscillogram
og.fill_between(times, data, color='k') 
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
# Spectrogram
sg.specgram(data, Fs=samplerate)
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('freq')
##plt.show()
plt.savefig('prefilter.png', dpi=100)
# Export Plot (postfilter)
fig, (og, sg) = plt.subplots(nrows=2)
# Oscillogram
og.fill_between(times, data2, color='k') 
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
# Spectrogram
sg.specgram(data, Fs=samplerate)
plt.xlim(times[0], times[-1])
plt.ylim(fcl-(fcu-fcl)*.25,fcu+(fcu-fcl)*.25)
##plt.ylim(fcl*0.5,fcu*1.25)
plt.xlabel('time (s)')
plt.ylabel('freq')
##plt.show()
plt.savefig('postfilter.png', dpi=100)



