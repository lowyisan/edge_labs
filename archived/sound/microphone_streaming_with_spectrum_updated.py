#!/usr/bin/env python3
"""
Top-Level Explanation:
This script demonstrates real-time audio acquisition and spectral analysis using FFT (Fast Fourier Transform).
It records audio from a microphone in chunks (frames), computes the FFT to obtain the frequency spectrum, and
displays both the raw audio waveform and its spectrum using Matplotlib. The script also measures the execution
time for the FFT computation, which can be useful for performance evaluation in real-time processing applications.
"""

#%% Import Required Libraries
import sounddevice as sd         # For capturing audio from the microphone.
                                # Q: Why use sounddevice?
                                # A: It provides an easy interface for real-time audio recording and playback.
import numpy as np               # For efficient numerical operations and array handling.
import matplotlib.pyplot as plt  # For plotting and visualizing the waveform and spectrum.
import time                    # For measuring execution time of the FFT process.

#%% Parameters
BUFFER = 1024 * 16           # Number of samples per frame. Adjust to change time resolution.
CHANNELS = 1                 # Use a single audio channel (mono recording).
RATE = 44100                 # Sampling rate in samples per second.
RECORD_SECONDS = 30          # Total duration for recording in seconds.
                                # Q: How does BUFFER size affect the analysis?
                                # A: A larger BUFFER increases frequency resolution but may add latency in real-time updates.

#%% Create Matplotlib Figure and Axes
# Set up a figure with two subplots: one for the time-domain waveform and one for the frequency spectrum.
fig, (ax1, ax2) = plt.subplots(2, figsize=(4, 4))

# Generate x-axis values for the waveform (sample indices)
x = np.arange(0, 2 * BUFFER, 2)

# Compute frequency bins for the FFT. Only the first half is needed due to symmetry.
xf = np.fft.fftfreq(BUFFER, 1 / RATE)[:BUFFER // 2]

# Create placeholder line objects for waveform and spectrum plots with random initial data.
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)
line_fft, = ax2.plot(xf, np.random.rand(BUFFER // 2), '-', lw=2)

# Format the waveform subplot.
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('Samples')
ax1.set_ylabel('Volume')
ax1.set_ylim(-5000, 5000)  # Adjust the y-axis limits based on expected amplitude levels.
ax1.set_xlim(0, BUFFER)

# Format the spectrum subplot.
ax2.set_title('SPECTRUM')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Log Magnitude')
ax2.set_ylim(0, 1000)      # Y-axis limits for the magnitude; adjust based on signal strength.
ax2.set_xlim(0, RATE / 2)  # X-axis covers frequencies from 0 to the Nyquist frequency.

# Display the plot in non-blocking mode to allow real-time updates.
plt.show(block=False)

#%% Recording Audio and Constructing the Spectrum
exec_time = []  # List to store execution times for FFT computation.

# Calculate the total number of frames based on the sampling rate, BUFFER size, and recording duration.
num_frames = (RATE // BUFFER) * RECORD_SECONDS

for _ in range(num_frames):
    # Record an audio frame from the microphone in int16 format.
    data = sd.rec(frames=BUFFER, samplerate=RATE, channels=CHANNELS, dtype='int16', blocking=True)
    data = np.squeeze(data)  # Remove any extra dimensions.
    
    # Compute the FFT of the recorded frame.
    start_time = time.time()  # Start timing the FFT computation.
    fft_data = np.fft.fft(data)
    # Take the magnitude of the FFT and keep only the positive frequencies.
    fft_data = np.abs(fft_data[:BUFFER // 2])
    exec_time.append(time.time() - start_time)  # Record the execution time.
    
    # Scale FFT data (e.g., multiply by a factor for normalization) and update the spectrum plot.
    line.set_ydata(data)  # Update the waveform plot with the new audio data.
    line_fft.set_ydata(2.0 / BUFFER * fft_data)  # Update the FFT plot.
    
    # Redraw the canvas to display updated plots.
    fig.canvas.draw()
    fig.canvas.flush_events()
    
#%% End of Streaming and Performance Reporting
print('Stream stopped')
# Calculate and print the average execution time of the FFT process in milliseconds.
print('Average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))

# Additional Q&A:
# Q: What does FFT do in this context?
# A: FFT converts the time-domain audio signal into its frequency components, allowing us to analyze the spectrum.
# Q: Why do we only use the first half of the FFT output?
# A: Because the FFT of a real-valued signal is symmetric, the second half contains redundant information.
