# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:06:36 2019

@author: Yems
"""

from scipy.io import wavfile
import os
from scipy import fftpack
import numpy as np

# Define sound file of interest
path = 'F:\\SS_DATA\\SS_BBF_beta'
sound = '5D2005C8.WAV'
samplerate, W = wavfile.read(os.path.join(path, sound))
L=len(W)

F=fftpack.fft(W)/len(W)
F=F[0:int(L/2)].astype(float)

F2=F*0;
F2[320:84000]=F[np.linspace(156320,240000-1,83680).astype(int)]

F2=np.concatenate((F2,F2[np.arange(len(F)-1,-1,-1)]))
W2=fftpack.ifft(F2*len(F)).astype(np.int16)

wavfile.write('transpose_og.WAV',samplerate,W)
wavfile.write('transpose_hi2lo.WAV',samplerate,W2)
