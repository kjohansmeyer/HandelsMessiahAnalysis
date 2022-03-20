import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sounddevice as sd

# Note: Headphones may be needed to hear sounds. Sample rate may be too high 
# for some devices (laptop/tablet speakers).

# ----------------------------- Provided Code ------------------------------- # 
music = loadmat('Handel.mat')
t = music['x'][0]
S = music['v'][0]
# plt.figure(1) # added to provided code
# plt.plot(t,S) # plot signal in the time domain
# plt.title("Handel's Messiah")
# plt.ylabel('Intensity')
# plt.xlabel('Time (s)')
# plt.show() # added to provided code

# ---------------------------- Play Audio Clip ------------------------------ #
sampleRate = len(S)/max(t) # ~9 second audio clip

# # Citation: https://python-sounddevice.readthedocs.io/en/0.4.4/usage.html#playback
# sd.play(S,samplerate=sampleRate) # uncomment to hear audio

# Observation: The audio appears to be playing at the proper sampling rate.
# There are definitely audible noise and pops. I am curious to compare the
# original and filtered audio clips/spectrograms.

# ------------------- Spectrogram using Gaussian Filter --------------------- #

a = 10 # Gaussian window width
NFFTwindow = np.arange(0,1,0.01)
b = max(NFFTwindow)/2
gaussianWindow = max(S)*np.exp(-a*(NFFTwindow-b)**2) 
# max(S) not needed, just to scale different. Spectrogram result is the same.

plt.figure()
plt.plot(t,S)
plt.title("Handel's Messiah")
plt.ylabel('Intensity')
plt.xlabel('Time (s)')
plt.plot(NFFTwindow,gaussianWindow)
plt.show()

plt.figure()
plt.specgram(S,Fs=sampleRate,NFFT=len(NFFTwindow),window=gaussianWindow,noverlap=0)
plt.title('Gaussian Filter, a = 10, Overlap = 0')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.show()

# --------- Spectrogram using Gaussian Filter (adjusting overlap) ----------- #

a = 10 # Gaussian window width
NFFTwindow = np.arange(0,1,0.01)
b = max(NFFTwindow)/2
gaussianWindow = max(S)*np.exp(-a*(NFFTwindow-b)**2) 
# max(S) not needed, just to scale different. Spectrogram result is the same.

plt.figure()
plt.plot(t,S)
plt.title("Handel's Messiah")
plt.ylabel('Intensity')
plt.xlabel('Time (s)')
plt.plot(NFFTwindow,gaussianWindow)
plt.show()

plt.figure()
plt.specgram(S,Fs=sampleRate,NFFT=len(NFFTwindow),window=gaussianWindow,noverlap=50)
plt.title('Gaussian Filter, a = 10, Overlap = 50 (oversampled) [zoomed]')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.ylim([0,100])
plt.xlim([0,0.5])
plt.show()

plt.figure()
plt.specgram(S,Fs=sampleRate,NFFT=len(NFFTwindow),window=gaussianWindow,noverlap=-50)
plt.title('Gaussian Filter, a = 10, Overlap = -50 (undersampled) [zoomed]')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.ylim([0,100])
plt.xlim([0,0.5])
plt.show()

# -------------------- Spectrogram using Ricker Wavelet --------------------- #

sigma = 0.01
NFFTwindow = np.arange(0,2,0.01)
b = max(NFFTwindow)/2
rickerWindow = (2/np.sqrt(3*sigma)*np.pi**(1/4))*(1-((NFFTwindow-b)/sigma)**2)*np.exp((-(NFFTwindow-b)**2)/(2*sigma**2))

plt.figure(20)
plt.plot(t,S)
plt.title("Handel's Messiah")
plt.ylabel('Intensity')
plt.xlabel('Time (s)')
plt.plot(NFFTwindow,rickerWindow)
plt.show()

plt.figure()
plt.specgram(S,Fs=sampleRate,NFFT=len(NFFTwindow),window=rickerWindow,noverlap=0)
plt.title('Ricker Wavelet Filter, σ = 0.01, Overlap = 0')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.show()

# # -------------- Creating Spectrogram from Gabor Transform ------------------ #
plt.figure(3)
plt.specgram(S,Fs=sampleRate)
plt.title('Spectrogram - Audio with Gabor Transform')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar()
plt.show()

# # -------------------- Creating Gabor Transform GIF ----------------------- #
# # I don't recommend uncommenting this section - The loop creates 93 images 
# # that are stitched together into a GIF.

# a = 10 # filter width
# dt = 0.1

# for i in range(0,93):
#     gaborFilter = np.exp(-a*(t-i*dt)**2)
#     filteredS = S*gaborFilter
#     plt.figure(i+100)
#     plt.grid('True')
#     plt.title('Gabor Transform of Audio Signal (a = 10)')
#     plt.ylabel('Intensity')
#     plt.xlabel('Time (s)')
#     plt.xlim((0,9))
#     plt.ylim((-0.25,0.25))
#     plt.plot(t,filteredS)
#     plt.savefig('PUT DIRECTORY HERE DIRECTORY' + "plot%i.png" % i)


# # ----------------- Filter All Intensities LESS THAN 100 -------------------- #

# # Citation: https://pages.mtu.edu/~suits/notefreq440.html
pianoNotes440 = [16.35,17.32,18.35,19.45,20.6,21.83,23.12,24.5,25.96,27.5,
                 29.14,30.87,32.7,34.65,36.71,38.89,41.2,43.65,46.25,49,51.91,
                 55,58.27,61.74,65.41,69.3,73.42,77.78,82.41,87.31,92.5,98,
                 103.83,110,116.54,123.47,130.81,138.59,146.83,155.56,164.81,
                 174.61,185,196,207.65,220,233.08,246.94,261.63,277.18,293.66,
                 311.13,329.63,349.23,369.99,392,415.3,440,466.16,493.88,
                 523.25,554.37,587.33,622.25,659.25,698.46,739.99,783.99,
                 830.61,880,932.33,987.77,1046.5,1108.73,1174.66,1244.51,
                 1318.51,1396.91,1479.98,1567.98,1661.22,1760,1864.66,1975.53,
                 2093,2217.46,2349.32,2489.02,2637.02,2793.83,2959.96,3135.96,
                 3322.44,3520,3729.31,3951.07,4186.01,4434.92,4698.63,4978.03,
                 5274.04,5587.65,5919.91,6271.93,6644.88,7040,7458.62,7902.13]

pianoNotes444 = [16.5,17.48,18.52,19.62,20.79,22.03,23.33,24.72,26.19,27.75,
                 29.4,31.15,33,34.96,37.04,39.24,41.58,44.05,46.67,49.44,52.39,
                 55.5,58.8,62.3,66,69.93,74.08,78.49,83.16,88.1,93.34,98.89,
                 104.77,111,117.6,124.59,132,139.85,148.17,156.98,166.31,176.2,
                 186.68,197.78,209.54,222,235.2,249.19,264,279.7,296.33,313.96,
                 332.62,352.4,373.36,395.56,419.08,444,470.4,498.37,528.01,
                 559.4,592.67,627.91,665.25,704.81,746.72,791.12,838.16,888,
                 940.8,996.75,1056.02,1118.81,1185.34,1255.82,1330.5,1409.61,
                 1493.43,1582.24,1676.32,1776,1881.61,1993.49,2112.03,2237.62,
                 2370.67,2511.64,2660.99,2819.22,2986.86,3164.47,3352.64,3552,
                 3763.21,3986.99,4224.06,4475.24,4741.35,5023.29,5321.99,
                 5638.45,5973.73,6328.94,6705.28,7104,7526.43,7973.97]

#               C     C#     D        D#        E       F       F#      G     G#       A       A#       B
noteColors = ['red','red','orange','orange','yellow','green','green','blue','blue','indigo','indigo','violet']

dt = 1/sampleRate
halfWayIndex = int(len(S)/2)
freq = (1/(dt*len(S)))*np.arange(halfWayIndex)

FFT_S = np.fft.fft(S)
modFFT_S = np.abs(FFT_S)

plt.figure(4)
plt.plot(freq,modFFT_S[0:halfWayIndex],zorder=2)
plt.title('Fourier Transform with Piano Notes (A: 440 Hz)')
plt.ylabel('Intensity')
plt.xlabel('Frequency (Hz)')
for i in range(0,96):
    n = int(i/12) # used to cycle through 12 colors for notes
    plt.vlines(pianoNotes440[i],0,375,colors=noteColors[i-n*12])
plt.show()

plt.figure(8)
plt.hlines(100,0,max(freq),linestyles='dashed',colors='red',zorder=2)
plt.plot(freq,modFFT_S[0:halfWayIndex],zorder=1)
plt.title('Fourier Transform of Signal')
plt.ylabel('Intensity')
plt.xlabel('Frequency (Hz)')
plt.show()


filteredFreq = modFFT_S > 100
FFT_Sfiltered = filteredFreq * FFT_S
IFFT_Sfiltered = np.fft.ifft(FFT_Sfiltered)
plt.figure(5)
plt.plot(t,IFFT_Sfiltered)
plt.title("Handel's Messiah - Intensities Greater Than 100")
plt.ylabel('Intensity')
plt.xlabel('Time (s)')
plt.show()

# sd.play(np.real(IFFT_Sfiltered),samplerate=sampleRate) #uncomment to hear audio

# Observation: All the intensities above 100 seem to be the background 
# harmonies. Instead of hearing "Hallelujah!" in the foreground, you can only
# hear "Haaaaaaaa!". It is difficult to indicate the specific song solely on
# these frequencies. 

# # --------------- Filter All Intensities GREATER THAN 100 ------------------- #
dt = 1/sampleRate
halfWayIndex = int(len(S)/2)
freq = (1/(dt*len(S)))*np.arange(halfWayIndex)

FFT_S = np.fft.fft(S)
modFFT_S = np.abs(FFT_S)

plt.figure()
plt.hlines(100,0,max(S),linestyles='dashed',colors='red',zorder=10)
plt.plot(freq,modFFT_S[0:halfWayIndex],zorder=2)
plt.title('Fourier Transform - Original Audio')
plt.ylabel('Intensity')
plt.xlabel('Frequency (Hz)')
plt.show()

plt.figure()
filteredFreq = modFFT_S < 100
FFT_Sfiltered = filteredFreq * FFT_S
IFFT_Sfiltered = np.fft.ifft(FFT_Sfiltered)
plt.plot(t,IFFT_Sfiltered)
plt.title("Handel's Messiah - Intensities Less Than 100")
plt.ylabel('Intensity')
plt.xlabel('Time (s)')
plt.show()

# sd.play(np.real(IFFT_Sfiltered),samplerate=sampleRate)

# Observation: All the intensities below 100 seem to be the foreground sounds.
# While the background, ambient sounds are missing, it is very easy to indicate
# the song. 
