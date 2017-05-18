
# coding: utf-8

# In[16]:

from pylab import linspace, subplot, plot, title, xlabel, ylabel, grid, axis, savefig, clf
from numpy import arange, log10, sqrt, mean, sum
from scipy.io.wavfile import read
from scipy import fft, ceil


# In[17]:

def plot_vowel(vowel):
    filename = "tone" + vowel + '.wav'
    
    #read the file in question
    (Fs, x) = read(filename)
    
    #create window for analysis
    win = int(Fs/4)
    
    #take the FFT with specific window length
    x_fft = fft(x,win)
    
    #obtain magnitude of frequencies using abs
    #scale by the number of signal points
    #square signal
    sig_pow = (abs (x_fft) / len(x)) **2
    
    #array conatining frequencies for window
    Freqs = arange(0, win, 2) * (Fs / win)
    
    #use only left side of frequency power spectrum
    sig_pow = sig_pow[0:(int(len(sig_pow)/2))]
    
    #plot the time-domain window of samples
    
    #calculate length of the .wav file in seconds
    time = win/Fs
    #create evenly spaced numbers between [0:time]
    t = linspace (0, time, win)
    #plot signal x
    subplot (2,1,1)
    plot(t,x[0:win])
    title('Sound plot of ' + filename)
    xlabel('Time')
    ylabel('Amplitude')
    axis('tight')
    grid(True)
    
    #scale frequency array to kHz and plot the signal in dB
    subplot(2,1,2)
    plot(Freqs/win, 10*log10(sig_pow), color='r')
    xlabel('Frequency (kHz)')
    ylabel('Power (dB)')
    axis('tight')
    grid(True)
    
    savefig(filename[0:5] + "_plot.png")
    
    clf()


# In[ ]:



