
# coding: utf-8

# In[144]:

from numpy.fft import rfft
from scipy import arange, sin, pi, fft, real, imag, log10
from scipy.io.wavfile import write, read
from matplotlib.pyplot import figure, plot, subplot, axis, grid
from pylab import xticks, yticks, xlim, ylim, xlabel, ylabel, title

import sounddevice as sd


# In[145]:

#Create array for formant frequencies and amplitudes
#3. Use the formant frequencies given by JC Wells - refer to table provided in the lectures (slide 44 of Lecture 8)
#2. Different amplitudes based on the spectrograms of the American English Vowels (Ladeforged 2006:185-187)
#   from slide 45 of Lecture 8 and spectograms
ff = numpy.array([
        ('a', 740, 1180, 2640, 7, 4.5, 2.5),
        ('e', 600, 2060, 2840, 6, 2, 2),
        ('i', 360, 2220, 2960, 7, 4.5, 2.5),
        ('o', 380, 940, 2300, 5, 3, 0.5),
        ('u', 320, 920, 2200, 5, 3, 0.5)
    ],
                 dtype=numpy.dtype([('vowel', 'U4'), ('F1', '<i4'), ('F2', '<i4'), ('F3', '<i4'), 
                                    ('a1', '<f4'), ('a2', '<f4'), ('a3', '<f4')]))


# In[162]:

#2. Create a function that generates each simple formant tone
def write_tone(ffrow, duration=2): #duration 2 seconds (by default)

    vowel = ffrow[0] #name of vowel
    f1 = ffrow[1] #formant 1 frequency
    f2 = ffrow[2] #formant 2 frequency
    f3 = ffrow[3] #formant 3 frequency
    a1 = ffrow[4] #amplitude for formant 1
    a2 = ffrow[5] #amplitude for formant 2
    a3 = ffrow[6] #amplitude for formant 3
    
    #sampling frequency = Fs = 8 kHz = 8000 Hz
    Fs = 8000
    
    #create vector for samples = sampling frequency (samples/sec) * duration (seconds)
    a = arange(int(Fs*duration))

    #1. Synthesize each formant tone in preparation for creating each vowel
    #Constraint 1: use pure tones only using the sin function
    #Constraint 2: manually adjust the amplitude of each formant frequency
    #Constraint 3: use the first three formants for the average male speaker
    signal1 = a1*sin(2*pi*a*(f1/Fs))
    signal2 = a2*sin(2*pi*a*(f2/Fs))
    signal3 = a3*sin(2*pi*a*(f3/Fs))

    #4. Simply add all three formants for each vowel
    signal = signal1 + signal2 + signal3
    
    #4. and save the files in .wav format
    filename = 'tone' + vowel + '.wav'
    write(filename, Fs, signal)
        
    return filename, Fs


# In[161]:

#5. Extra point: incorporate the code provided in the previous slide as a module.py to plot each vowel sound
from module import plot_vowel

for i in range(len(ff)):
    
    write_tone(ff[i])
    
    #5. plot each vowel sound using module.py
    plot_vowel(ff[i][0])

