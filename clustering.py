# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:24:55 2019

@author: Yems
"""

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import os
from scipy.io import wavfile
from scipy import signal
import numpy
import mFeatExtract as mfe
#from playsound import playsound

# Parameters
FILT=1
fcl = 15000 #Cut-off freq. lower limit
fcu = 23000 #Cut-off freq. upper limit 

# Define sound file of interest
#DET 5D0A00B6
#active forest 5D3329A0
#quiet forest 5D34A0A0
path = 'C:\\Users\\Yems\\Desktop\\repos\\SS_sheperd'
#sound = '5D3329A0.WAV'
X = []
S = []
for sound in os.listdir(path):   
    if '.WAV' in sound:
        # Load the data and calculate the time of each sample
        samplerate, data = wavfile.read(os.path.join(path, sound))
        times = numpy.arange(len(data))/float(samplerate)
        ##data=data[:,1] #mono
        
        if FILT == 1:
            # Filter (working parameters
            w = fcl / (samplerate/2) #Normalize freq
            b, a = signal.butter(5,w,'high')
            temp = signal.filtfilt(b,a,data)
            w = fcu / (samplerate/2) #Normalize freq
            b, a = signal.butter(5,w,'low')
            data = signal.filtfilt(b,a,temp)
            
        mX = mfe.FeatArray(data,samplerate,0.1*samplerate, 0.025*samplerate)
        X.append(numpy.mean(mX,1)) 
        S.append(numpy.var(mX,1))
X=numpy.array(X)
S=numpy.array(S)

A=numpy.concatenate((X,S),axis=1)
A=StandardScaler().fit_transform(A)
# create kmeans object
kmeans = KMeans(n_clusters=4)
# fit kmeans object to data
kmeans.fit(A)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
# save new clusters for chart
g = kmeans.fit_predict(A)

# dimension reduction
pca = PCA(n_components=2)
PC = pca.fit_transform(A)

# Plot clusters
plt.scatter(PC[g ==0,0], PC[g == 0,1], s=1, c='magenta')
plt.scatter(PC[g ==1,0], PC[g == 1,1], s=1, c='cyan')
plt.scatter(PC[g ==2,0], PC[g == 2,1], s=1, c='orange')
plt.scatter(PC[g ==3,0], PC[g == 3,1], s=1, c='green')

#plt.scatter(list(range(len(g))), g, s=1)

numpy.savetxt(os.path.join(path, "clusters_g")+".csv", g, delimiter=",")
numpy.savetxt(os.path.join(path, "clusters_A")+".csv", A, delimiter=",")

# Plot clusters
x=0
y=1
plt.scatter(A[g ==0,x], A[g == 0,y], s=1, c='magenta')
plt.scatter(A[g ==1,x], A[g == 1,y], s=1, c='cyan')
plt.scatter(A[g ==2,x], A[g == 2,y], s=1, c='orange')
plt.scatter(A[g ==3,x], A[g == 3,y], s=1, c='green')

