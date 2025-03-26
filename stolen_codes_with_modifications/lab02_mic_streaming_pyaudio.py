#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script captures real-time audio from a microphone using PyAudio, performs a Fast Fourier Transform (FFT)
on the captured data to analyze its frequency components, and displays both the time-domain waveform and the 
frequency spectrum in real time using Matplotlib. It demonstrates key concepts such as audio data conversion,
FFT computation, and real-time plotting—useful for lab tests in audio signal processing and spectral analysis.

Potential Lab Test Questions:
Q1. What does the FFT (Fast Fourier Transform) provide in audio analysis?
   A1. The FFT converts a time-domain signal into its frequency-domain representation, showing the intensity 
       of various frequency components in the signal.
Q2. Why is it important to convert raw audio byte data into integers before performing an FFT?
   A2. Converting to integers (e.g., int16) allows numerical operations (like the FFT) to be performed correctly on the audio samples.
Q3. What do the x-axis limits in the spectrum plot represent?
   A3. The x-axis represents frequency, ranging from 0 Hz to the Nyquist frequency (half the sampling rate).
"""

#%% Import the required libraries

import pyaudio  # For audio input/output (real-time access to the microphone)
# See: https://people.csail.mit.edu/hubert/pyaudio/

import struct  # For converting raw audio byte data into numeric values (int16)
# See: https://docs.python.org/3/library/struct.html

import numpy as np  # For numerical operations and handling audio samples as arrays
import matplotlib.pyplot as plt  # For plotting waveforms and spectrum
from scipy.fftpack import fft, fftfreq  # For performing Fast Fourier Transform (FFT) and computing frequency bins
# See: https://docs.scipy.org/doc/scipy/tutorial/fft.html

import time  # For tracking time taken to compute FFT

#%% Parameters for audio stream and visualization

BUFFER = 1024 * 16           # Number of audio samples per frame (larger = more smoothing, more latency)
FORMAT = pyaudio.paInt16     # 16-bit audio format (standard for most microphones)
CHANNELS = 1                 # Mono audio input (one microphone)
RATE = 44100                 # Sample rate in Hz (CD quality audio)
RECORD_SECONDS = 30          # Duration of audio capture in seconds

#%% Set up initial plots for waveform and frequency spectrum

# Create a figure with two subplots: waveform (top) and spectrum (bottom)
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))

# X-axis for waveform plot: sample indices (0 to BUFFER with step of 2 since int16 = 2 bytes)
x = np.arange(0, 2 * BUFFER, 2)

# X-axis for spectrum plot: frequency bins from 0 Hz to Nyquist frequency (RATE/2)
xf = fftfreq(BUFFER, 1 / RATE)[:BUFFER // 2]

# Create placeholder lines for the waveform and spectrum plots (initialized with random values)
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)         # Waveform line for raw audio
line_fft, = ax2.plot(xf, np.random.rand(BUFFER // 2), '-', lw=2)  # Spectrum line for frequency analysis

# === Format Waveform Plot (Top) ===
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(-5000, 5000)   # Y-axis range based on expected amplitude
ax1.set_xlim(0, BUFFER)

# === Format Spectrum Plot (Bottom) ===
ax2.set_title('SPECTRUM')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Log Magnitude')
ax2.set_ylim(0, 1000)       # Y-axis range for intensity (adjust as needed)
ax2.set_xlim(0, RATE / 2)   # Frequency range from 0 Hz to Nyquist frequency

# Display the plots non-blocking (allows real-time updating)
plt.show(block=False)

#%% Initialize PyAudio input stream

audio = pyaudio.PyAudio()  # Create a PyAudio instance

# Open a stream to access the microphone with both input and optional output for monitoring
stream = audio.open(
    format=FORMAT,             # 16-bit audio format
    channels=CHANNELS,         # Mono channel
    rate=RATE,                 # Sampling rate
    input=True,                # Enable input (microphone)
    output=True,               # Enable output (optional for monitoring)
    frames_per_buffer=BUFFER   # Number of samples per chunk
)

print('stream started')

# Initialize list to store execution time for each FFT (for performance profiling)
exec_time = []

#%% Main loop: Capture audio, perform FFT, and update plots in real time

# Calculate the total number of chunks to be captured during RECORD_SECONDS
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):

    # === Read audio chunk from the microphone ===
    data = stream.read(BUFFER)  # Capture raw binary audio data

    # === Convert binary audio data to integers ===
    # 'h' indicates 16-bit signed integer; unpack BUFFER samples from the raw data
    data_int = struct.unpack(str(BUFFER) + 'h', data)

    # === Perform FFT to extract frequency components ===
    start_time = time.time()     # Start timer before FFT computation
    yf = fft(data_int)           # Compute FFT (returns complex numbers)
    exec_time.append(time.time() - start_time)  # Store FFT computation time

    # === Update plots in real time ===
    
    # Update the waveform plot with the latest audio data
    line.set_ydata(data_int)

    # Update the spectrum plot with the log magnitude of the FFT result
    # We take the absolute value of the FFT, scale it, and consider only the first half (positive frequencies)
    line_fft.set_ydata(2.0 / BUFFER * np.abs(yf[0:BUFFER // 2]))

    # Redraw the figure canvas to reflect the new data
    fig.canvas.draw()
    fig.canvas.flush_events()  # Ensure immediate update of GUI events

#%% Cleanup: Stop the audio stream and report performance

audio.terminate()  # Terminate the PyAudio stream to release resources

print('stream stopped')
# Print the average FFT computation time per frame in milliseconds
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))
# Q: What does the average execution time indicate?
# A: It measures the average time taken to compute the FFT on each frame, which is critical for real-time processing performance.
