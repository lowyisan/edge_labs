#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script captures real-time audio from a microphone using PyAudio, applies a digital bandpass filter,
and plots both the raw and filtered audio waveforms in real time. It demonstrates how to design and apply
a second-order sections (SOS) bandpass filter using SciPy, as well as how to perform real-time plotting with Matplotlib.
This is useful for lab tests on audio signal processing, filter design, and real-time visualization.

Potential Lab Test Questions:
Q1. What is the purpose of designing a bandpass filter in this script?
   A1. It allows frequencies within a specified range (here, around 19.4-19.6 kHz) to pass through while attenuating others.
Q2. Why is the Nyquist frequency used in filter design?
   A2. The Nyquist frequency (half the sampling rate) is used to normalize the cutoff frequencies for the digital filter.
Q3. How does real-time plotting work in this context?
   A3. The plots are updated in a loop using Matplotlib’s canvas redrawing methods to reflect the incoming audio data.
"""

#%% Import the required libraries
import pyaudio            # For real-time audio input/output from the microphone
# Reference: https://people.csail.mit.edu/hubert/pyaudio/

import struct             # For converting raw byte stream data into Python int16 format
# Reference: https://docs.python.org/3/library/struct.html

import numpy as np        # For numerical operations and array handling
import matplotlib.pyplot as plt  # For real-time plotting of audio waveforms
from scipy.signal import butter, sosfilt  # For designing and applying a digital bandpass filter
# Reference: https://docs.scipy.org/doc/scipy/reference/signal.html

import time               # For measuring execution time of the filter processing

#%% Audio and buffer parameters
BUFFER = 1024 * 16           # Number of audio samples per frame (chunk size)
FORMAT = pyaudio.paInt16     # Audio format: 16-bit PCM
CHANNELS = 1                 # Number of audio channels (Mono)
RATE = 44100                 # Sampling rate in Hz
RECORD_SECONDS = 20          # Duration of the real-time processing session in seconds

#%% Create the initial plot for waveform and filtered output
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))  # Create two subplots stacked vertically

# Create an x-axis array for sample indices
x = np.arange(0, 2 * BUFFER, 2)

# Initialize placeholder lines for the raw and filtered signals
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)         # Line for the raw waveform
line_filter, = ax2.plot(x, np.random.rand(BUFFER), '-', lw=2)    # Line for the filtered signal

# === Configure the waveform plot (raw audio) ===
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('Samples')
ax1.set_ylabel('Amplitude')
ax1.set_ylim(-5000, 5000)   # Set y-axis limits for amplitude display
ax1.set_xlim(0, BUFFER)     # Set x-axis limits based on buffer size

# === Configure the filtered signal plot ===
ax2.set_title('FILTERED')
ax2.set_xlabel('Samples')
ax2.set_ylabel('Amplitude')
ax2.set_ylim(-5000, 5000)
ax2.set_xlim(0, BUFFER)

# Display the plot in a non-blocking mode for real-time updates
plt.show(block=False)

#%% Function to design a digital bandpass filter
def design_filter(lowfreq, highfreq, fs, order=3):
    """
    Designs a bandpass filter using a Butterworth filter with second-order sections (SOS).

    Parameters:
    - lowfreq: Lower cutoff frequency in Hz.
    - highfreq: Upper cutoff frequency in Hz.
    - fs: Sampling rate in Hz.
    - order: Filter order (controls the steepness of the roll-off).

    Returns:
    - sos: Second-order sections representation of the filter.
    """
    nyq = 0.5 * fs                # Calculate Nyquist frequency (half the sampling rate)
    low = lowfreq / nyq           # Normalize the low cutoff frequency
    high = highfreq / nyq         # Normalize the high cutoff frequency
    sos = butter(order, [low, high], btype='band', output='sos')  # Design bandpass filter in SOS format
    return sos

# Design the bandpass filter with a passband around 19.4 kHz to 19.6 kHz.
# NOTE: The filter design uses fs = 48000, so ensure consistency with your actual sample rate.
sos = design_filter(19400, 19600, 48000, 3)
# Q: Why do we normalize frequencies using the Nyquist frequency?
# A: Normalization ensures that the cutoff frequencies are represented as a fraction of the Nyquist rate, which is required for digital filter design.

#%% Set up the PyAudio stream for real-time microphone input
audio = pyaudio.PyAudio()  # Initialize PyAudio

# Open the audio stream for both input (microphone) and output (monitoring, optional)
stream = audio.open(
    format=FORMAT,             # Audio format: 16-bit PCM
    channels=CHANNELS,         # Mono channel
    rate=RATE,                 # Sampling rate
    input=True,                # Enable input for capturing microphone data
    output=True,               # Enable output for potential playback (monitoring)
    frames_per_buffer=BUFFER   # Chunk size for each frame of audio
)

print('stream started')

# List to store the execution time for each frame processing (for performance profiling)
exec_time = []

#%% Real-time audio processing loop
# Loop for the total number of frames corresponding to RECORD_SECONDS
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):

    # === Read a chunk of audio data ===
    data = stream.read(BUFFER)  # Read raw bytes from the microphone

    # === Convert binary data to a NumPy int16 array ===
    data_int = struct.unpack(str(BUFFER) + 'h', data)  # Unpack the byte data into BUFFER int16 samples
    
    # === Apply bandpass filter ===
    start_time = time.time()         # Start timer to measure filter execution time
    yf = sosfilt(sos, data_int)        # Apply the designed filter using second-order sections
    exec_time.append(time.time() - start_time)  # Log the time taken to filter the data

    # === Update plots with the real-time data ===
    line.set_ydata(data_int)    # Update raw waveform plot with current audio data
    line_filter.set_ydata(yf)   # Update filtered signal plot with processed data
    fig.canvas.draw()           # Redraw the figure with updated data
    fig.canvas.flush_events()   # Flush GUI events to update the plot in real time

#%% Terminate the audio stream after the recording session is complete
audio.terminate()  # Properly close the audio stream

print('stream stopped')
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))
# Q: Why is it important to record and print the execution time?
# A: It helps assess the performance of the filtering process and ensures that the real-time constraints are met.
