#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script demonstrates real-time audio capture and processing using SoundDevice,
NumPy, Matplotlib, and SciPy. It records audio from the microphone, applies a digital
bandpass filter (using a Butterworth design) to isolate a narrow frequency range (19.4–19.6 kHz),
and updates live plots showing both the raw and filtered audio waveforms. This example is
useful for lab tests on digital signal processing, real-time visualization, and filter design.

Potential Lab Test Questions:
Q1. What is the purpose of a bandpass filter in this context?
   A1. The bandpass filter isolates a narrow frequency band from the audio signal, allowing analysis
       of specific frequency components while attenuating others.
Q2. How is real-time audio captured and processed in this script?
   A2. The script uses SoundDevice to capture audio in chunks (frames), applies the filter to each frame,
       and updates live plots with Matplotlib to visualize the results.
Q3. Why is it necessary to normalize cutoff frequencies using the Nyquist frequency?
   A3. Normalization ensures that the filter design operates correctly relative to the sampling rate,
       as the Nyquist frequency represents half the sample rate.
"""

#%% Import the required libraries
import sounddevice as sd  # For real-time audio input/output from the microphone
# Reference: https://python-sounddevice.readthedocs.io/en/0.4.6/

import numpy as np  # For numerical processing of audio data
import matplotlib.pyplot as plt  # For real-time plotting of waveforms
import time  # For measuring time taken to process each frame

from scipy.signal import butter, sosfilt  # For designing and applying digital bandpass filters
# Reference: https://docs.scipy.org/doc/scipy/reference/signal.html

#%% Parameters for audio recording and buffer
BUFFER = 1024 * 16           # Number of samples to capture per frame (chunk size)
CHANNELS = 1                 # Mono recording (1 microphone)
RATE = 44100                 # Audio sample rate in Hz (CD quality)
RECORD_SECONDS = 20          # Total recording duration in seconds

#%% Set up Matplotlib figure and live plots
# Create a figure with two vertically stacked subplots: one for raw audio, one for filtered audio
fig, (ax1, ax2) = plt.subplots(2, figsize=(4, 4))

# Generate x-axis values (sample indices) for time-domain plots
x = np.arange(0, 2 * BUFFER, 2)

# Initialize both plots with random placeholder data
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)         # Plot for raw audio waveform
line_filter, = ax2.plot(x, np.random.rand(BUFFER), '-', lw=2)    # Plot for filtered audio waveform

# === Configure raw waveform plot ===
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('amplitude')
ax1.set_ylim(-5000, 5000)   # Set Y-axis limits for amplitude display
ax1.set_xlim(0, BUFFER)     # Set X-axis limits based on the buffer size

# === Configure filtered waveform plot ===
ax2.set_title('FILTERED')
ax2.set_xlabel('samples')
ax2.set_ylabel('amplitude')
ax2.set_ylim(-5000, 5000)
ax2.set_xlim(0, BUFFER)

# Display the figure window without blocking code execution (for real-time updates)
plt.show(block=False)

#%% Define a bandpass filter using Butterworth design
def design_filter(lowfreq, highfreq, fs, order=3):
    """
    Creates a digital bandpass filter using a Butterworth design with second-order sections (SOS).

    Parameters:
        lowfreq (float): Lower cutoff frequency in Hz.
        highfreq (float): Upper cutoff frequency in Hz.
        fs (int): Sampling rate in Hz.
        order (int): Filter order; higher order provides a steeper roll-off but requires more processing.

    Returns:
        sos (ndarray): Second-order sections representation of the bandpass filter.
    """
    nyq = 0.5 * fs               # Calculate the Nyquist frequency (half the sample rate)
    low = lowfreq / nyq          # Normalize the low cutoff frequency
    high = highfreq / nyq        # Normalize the high cutoff frequency
    sos = butter(order, [low, high], btype='band', output='sos')  # Create the bandpass filter in SOS format
    return sos

# Design a bandpass filter to isolate 19.4–19.6 kHz
# Ensure that the filter's sample rate (fs) matches RATE (here 44100 Hz)
sos = design_filter(19400, 19600, 44100, 3)
# Q: Why do we normalize the cutoff frequencies using the Nyquist frequency?
# A: Normalization scales the cutoff frequencies relative to the Nyquist limit, ensuring proper filter design in the digital domain.

#%% Real-time audio processing loop
exec_time = []  # List to store processing time per frame (for performance profiling)

# Calculate the total number of frames to capture during the recording session
num_frames = (RATE // BUFFER) * RECORD_SECONDS

for _ in range(0, num_frames):
    # === Record a chunk of audio ===
    # Blocking mode waits until the full BUFFER is captured before continuing
    data = sd.rec(frames=BUFFER, samplerate=RATE, channels=CHANNELS, dtype='int16', blocking=True)

    # Remove singleton dimension if present (e.g., [BUFFER, 1] -> [BUFFER])
    data = np.squeeze(data)

    # === Apply the bandpass filter ===
    start_time = time.time()         # Record start time for filtering
    yf = sosfilt(sos, data)           # Filter the audio data using the designed bandpass filter
    exec_time.append(time.time() - start_time)  # Log the execution time for filtering

    # === Update the plots ===
    line.set_ydata(data)             # Update the raw audio waveform plot with current data
    line_filter.set_ydata(yf)        # Update the filtered audio waveform plot with processed data
    fig.canvas.draw()                # Redraw the figure with the new data
    fig.canvas.flush_events()        # Ensure immediate GUI updates

#%% Cleanup and report
print('stream stopped')  # Notify that the audio capture has ended
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))
# Q: What does the execution time measurement indicate?
# A: It shows the average time taken to apply the filter to each frame, which is important for ensuring that
#    the processing meets real-time performance constraints.
