import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
import struct
from scipy.io import wavfile
import os
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm
import sklearn
import pandas as pd
from scipy.fftpack import fft,fftfreq
from scipy import signal
import math
from scipy.signal import butter, lfilter, freqz

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='hp', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter requirements.
order = 6
fs = 48000    # sample rate, Hz
cutoff = 150  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_highpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, cutoff * 2)
plt.title(f"Highpass Filter Frequency Response for Order = {order}")
plt.xlabel('Frequency [Hz]')
plt.grid()
plt.show()

############################################################################

samplerate, data = wavfile.read(f'{rel_dir}/{item}')
        
        #### ALL CODE BELOW IS DEDICATED TO FEATURE ENGINEERING
        
        # trim first 60k samples (shortly after clap)
        data = data[60000:]
        
        
        # mean center and scale data
        data = (data - data.mean()) / data.std()
        
        # filter using HPF
        if( en_filter):
            data = butter_highpass_filter(data, cutoff, fs, order);
        data = np.abs(data)

        # compresses data into bins to reduce dimentionality, works great after filter
        data = data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)
                
        # looks at the data for the first non-zero value (first peak in wave)
        first_peak = -1
        for i in range(data.size):
            if( data[i] != 0 and first_peak == -1):
                first_peak = i
        #trim from first peak onwards to end (sound runs long anways)
        data = data[first_peak:]

        # trims from end to have normalized size, scaled by width to accomidate different binning
        data = data[:int(trim_value/width)]
        
        # If peak value < 0.5 must be ambient noise
        #print(f"{np.max(data)} {item}")
        if ( np.max(data) < 0.36 and en_filter):
            data[:] = 0
            #print(item)
        
        x.append(data)
            
        
            
    x = np.array(x)