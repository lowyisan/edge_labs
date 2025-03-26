#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script captures real-time audio using SoundDevice, processes each audio chunk with a Fast Fourier Transform (FFT)
to extract frequency components, and visualizes both the time-domain waveform and the frequency spectrum in real time
using Matplotlib. The example demonstrates how to record audio, compute its FFT for spectral analysis, and update
live plots. This is useful for lab tests covering topics like real-time signal processing and audio analysis.

Potential Lab Test Questions:
Q1. What is the purpose of using FFT in audio processing?
   A1. The FFT converts time-domain signals into the frequency domain, revealing the magnitude of frequency components
       present in the audio.
Q2. How do you interpret the x-axis in the frequency spectrum plot?
   A2. The x-axis represents frequency bins up to the Nyquist frequency (half of the sampling rate), showing the frequency
       content of the audio signal.
Q3. Why is real-time plotting important in this context?
   A3. Real-time plotting allows you to visualize the live audio waveform and spectrum as the audio is captured and processed.
"""

#%% Import the required libraries

import sounddevice as sd  # For audio input/output using Python (easy and cross-platform)
# See: https://python-sounddevice.readthedocs.io/en/0.4.6/

import numpy as np  # For numerical processing and FFT computation
import matplotlib.pyplot as plt  # For real-time plotting of waveform and spectrum
import time  # For timing the FFT computation

#%% Parameters for audio and recording

BUFFER = 1024 * 16           # Number of samples per frame (chunk size). Larger = smoother but more delay
CHANNELS = 1                 # Mono recording (1 microphone)
RATE = 44100                 # Audio sample rate (samples per second)
RECORD_SECONDS = 30          # Duration of the recording session (in seconds)

#%% Setup matplotlib figure and line plots

# Create a figure with 2 vertically stacked subplots: waveform (top) and frequency spectrum (bottom)
fig, (ax1, ax2) = plt.subplots(2, figsize=(4, 4))

# Generate X-axis values for waveform: sample indices (every 2 for 16-bit audio)
x = np.arange(0, 2 * BUFFER, 2)

# Generate X-axis values for frequency spectrum: frequency bins (up to Nyquist frequency)
xf = np.fft.fftfreq(BUFFER, 1 / RATE)[:BUFFER // 2]

# Initialize line plots with random data (placeholders)
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)            # Waveform plot line
line_fft, = ax2.plot(xf, np.random.rand(BUFFER // 2), '-', lw=2)  # Spectrum plot line

# === Configure waveform axis ===
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(-5000, 5000)  # Adjust based on expected volume range
ax1.set_xlim(0, BUFFER)

# === Configure frequency spectrum axis ===
ax2.set_title('SPECTRUM')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Log Magnitude')
ax2.set_ylim(0, 1000)      # Adjust based on expected intensity
ax2.set_xlim(0, RATE / 2)  # Show up to Nyquist frequency

# Display the figure without blocking (so it can update in real time)
plt.show(block=False)

#%% Start recording loop and update plots with real-time audio

exec_time = []  # List to record execution times for FFT processing

# Determine number of chunks to process based on total duration and buffer size
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):

    # === Record a single chunk of audio ===
    # 'blocking=True' ensures the function waits until the full BUFFER is captured
    data = sd.rec(frames=BUFFER, samplerate=RATE, channels=CHANNELS, dtype='int16', blocking=True)
    data = np.squeeze(data)  # Remove singleton dimension (shape: [BUFFER])

    # === Perform FFT to get frequency domain representation ===
    start_time = time.time()  # Start timer to measure FFT performance
    fft_data = np.fft.fft(data)  # Compute FFT of the audio chunk (complex output)
    fft_data = np.abs(fft_data[:BUFFER // 2])  # Take magnitude of positive frequencies only

    # Record FFT processing time
    exec_time.append(time.time() - start_time)

    # === Update the plots in real-time ===

    # Update waveform plot with new audio data
    line.set_ydata(data)

    # Update spectrum plot with scaled FFT magnitude
    line_fft.set_ydata(2.0 / BUFFER * fft_data)

    # Redraw the updated figure
    fig.canvas.draw()
    fig.canvas.flush_events()

#%% Post-recording summary and cleanup

print('stream stopped')  # Indicate end of recording session
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))
# Q: What does the average execution time indicate?
# A: It shows the average time taken to compute the FFT per frame, which is important for assessing real-time performance.
