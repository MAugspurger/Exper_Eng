###import libraries needed

from scipy import signal 
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftshift
from scipy.io import wavfile
import scipy.io.wavfile as wavf
from playsound import playsound

rate, audio = wavfile.read('C_trumpet_E4.wav') ###reads in trumpet, rate = sampling rate in Hz, audio is the y values


N = audio.shape[0]  ###gives the length of the array

dt=1/rate  ### in sec, basically the definition of sampling rate, the higher, the smaller the dt, the more freqs (better quality) the sound (although there is no need for the rate to be > 2*20,000 Hz since our ears can't hear above 20,000; the 2 has to do with Nyquist sampling theorem)
x=np.linspace(-N/2,N/2-1,N)  ### make an index that goes from -half the pts to +half the pts; center is 0 now
totaltime=N/rate  ##in sec
df=1/totaltime   ###in Hz, important FFT relation when defining Fourier variables: dt=1/totalf, df=1/totaltime, df = totalf/N, dt= totaltime/N
time=dt*x  ##makes an array of elapsed time, centered at 0
freq=df*x  ##Makes an array of frequency offset, centered at 0

####plot the audio file as the waveform, should match trace on wavepad
plt.figure()
plt.plot(time, audio)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude)')
#plt.xlim(100, 101)  ###  uncomment this line and play with this to see inside the envelope, make this range smaller

####Spectrogram Function
M=1024  ###takes slices of 1024 pts to FFT
freqs, times, Sx = signal.spectrogram(audio, fs=rate, window='hanning',
                                      nperseg=1024, noverlap=M - 100,
                                      detrend=False, scaling='spectrum')


####cuts up audio into 1024 slices and stores them in a 3D structure: freqs, times, and Sx (which is the strength), then overlaps the edges by 100 pts and smooths the edges with a Hanning window so not choppy

###alternative plot commands for a surface plot
f, ax = plt.subplots(figsize=(4.8, 2.4))
ax.pcolormesh(times, freqs / 1000, 10*np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]');

plt.ylim(0,5)


###now let's do some Fourier magic!!!

# #######Taking Fourier transform of entire file shows the frequencies present for the entire duration of the clip ~7 seconds
yf=fftshift(fft(audio))

####plot FFT, each spike represents a sine wave; if a spike is negative, that has to do with the phase of that sinewave component
plt.figure()
plt.plot(freq/1000, yf)    ###dividing by 1000 here makes the scale kHz
plt.xlabel('Frequency (kHz)')
plt.ylabel('Amplitude)')
plt.xlim(-2,2)


####make simple LPF
####Because I am using FFt (operates in complex space) we have both positive and negative freqs
####an LPF  is going to be what we drew on wavepad, but reflected about the zero, HPF will just be the flip of the LPF (1's go to 0's, 0's go to 1's)

# ####filter
yfilt=np.zeros(len(freq)) ###just makes an array of all zeros that's the sam length as signal
fedge =500  ### edge of low pass in Hz
edge=fedge/df   ####converts real freqs to  #pts in the array so python knows what I mean by 500 Hz
rangelo=round(N/2-edge) ##fedge Hz lower than middle
rangehi=round(N/2+edge) ###fedge Hz higher than middle
yfilt[rangelo:rangehi]=1  ###takes the array of all zeros and puts 1's now from rangelo to rangehi

####replot FFT and the filter function to see if we matched what we wanted

plt.figure()
plt.plot(freq/1000, yf/max(yf)) #now Im normalizing spectrum so we can see the filter function on the same plot/scale
plt.plot(freq/1000,yfilt,'-r')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Amplitude)')
plt.xlim(-2,2)

###time to do the filtering, just a simple multiplication in the freq domain
yfnew=yf*yfilt

###plot filtered spectrum; should be just what we let in
plt.figure()
plt.plot(freq/1000, yfnew/max(yfnew))
plt.plot(freq/1000,yfilt,'-r')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Amplitude)')
plt.xlim(-2,2)


####transform back into time domain with inverse fft or ifft
yt=ifft(fftshift(yfnew))

###Filtered wav in time domain
plt.figure()
plt.plot(time, np.real(yt),'-r')
plt.plot(time, audio, '--')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude)')
plt.xlim(0,0.02)   #Zoom up of before and after; play with this range to look at the wave

audiout=np.real(yt)   ### our yt was actually complex, just take real part
audiout=np.asarray(audiout, dtype=np.int16)  ###this helps convert it to int16 so we can write back to .wav


####Spectrogram Filtered wave
M=1024  ###takes slices of 1024 pts to FFT
freqs, times, Sx = signal.spectrogram(audiout, fs=rate, window='hanning',
                                      nperseg=1024, noverlap=M - 100,
                                      detrend=False, scaling='spectrum')


####cuts up audio into 1024 slices and stores them in a 3D structure: freqs, times, and Sx (which is the strength), then overlaps the edges by 100 pts and smooths the edges with a Hanning window so not choppy

###alternative plot commands for a surface plot
f, ax = plt.subplots(figsize=(4.8, 2.4))
ax.pcolormesh(times, freqs / 1000, 10*np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]');

plt.ylim(0,5)

###finally, write wav to playable file
wavf.write('audin.wav', rate, audio) ###writes input in format python playsound can play, redundant
wavf.write('audiout.wav', rate, audiout) #we killed a lot of energy so may need to crank it up

###uncomment and run in command line or with f9 to play sound in python (no need to use vlc, or mediplayer, etc)
#playsound('audin.wav')
#playsound('audiout.wav')
