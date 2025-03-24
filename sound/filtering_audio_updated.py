#!/usr/bin/env python3
"""
Top-Level Explanation:
This script demonstrates real-time audio acquisition, processing, and visualization using Python.
It captures audio from a microphone in real time, applies a bandpass filter to each audio frame, and then
displays both the raw and filtered audio waveforms using Matplotlib. Additionally, it measures the execution
time of the filtering process to help evaluate performance.
"""

#%% Import Required Libraries
import sounddevice as sd         # For capturing audio from the microphone.
                                # Q: Why use sounddevice?
                                # A: It provides a simple interface for real-time audio input/output.
import numpy as np               # For numerical operations and array handling.
import matplotlib.pyplot as plt  # For plotting and visualizing audio waveforms.
import time                    # To measure execution time of the filtering process.
from scipy.signal import butter, sosfilt  
                                # For designing and applying a bandpass filter.
                                # Q: What are butter and sosfilt used for?
                                # A: 'butter' designs a Butterworth filter, and 'sosfilt' applies the filter in second-order sections.

#%% Parameters
BUFFER = 1024 * 16           # Number of samples per frame. Adjust to change time resolution.
CHANNELS = 1                 # Single channel for mono audio recording.
RATE = 44100                 # Sampling rate in samples per second.
RECORD_SECONDS = 20          # Duration of the recording in seconds.
                                # Q: How does BUFFER affect the real-time performance?
                                # A: A larger BUFFER provides more samples per frame but can increase latency.

#%% Create Matplotlib Figure and Axes
# Set up a figure with two subplots for raw and filtered waveforms.
fig, (ax1, ax2) = plt.subplots(2, figsize=(4, 4))

# Generate x-axis values corresponding to sample indices.
x = np.arange(0, 2 * BUFFER, 2)  # The factor of 2 ensures proper spacing for visualization.

# Initialize line objects with random data as placeholders.
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)
line_filter, = ax2.plot(x, np.random.rand(BUFFER), '-', lw=2)

# Formatting for the raw audio waveform subplot.
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('Samples')
ax1.set_ylabel('Amplitude')
ax1.set_ylim(-5000, 5000)  # Adjust the y-axis range to suit expected amplitude variations.
ax1.set_xlim(0, BUFFER)

# Formatting for the filtered waveform subplot.
ax2.set_title('FILTERED')
ax2.set_xlabel('Samples')
ax2.set_ylabel('Amplitude')
ax2.set_ylim(-5000, 5000)
ax2.set_xlim(0, BUFFER)

# Display the plot in interactive mode.
plt.show(block=False)

#%% Function for Designing the Bandpass Filter
def design_filter(lowfreq, highfreq, fs, order=3):
    """
    Designs a Butterworth bandpass filter.
    
    Parameters:
        lowfreq (float): Lower cutoff frequency in Hz.
        highfreq (float): Upper cutoff frequency in Hz.
        fs (int): Sampling frequency in Hz.
        order (int): Order of the filter (default is 3).
    
    Returns:
        sos (ndarray): Second-order sections representation of the filter.
    """
    nyq = 0.5 * fs            # Nyquist frequency.
    low = lowfreq / nyq       # Normalized lower frequency.
    high = highfreq / nyq     # Normalized upper frequency.
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

# Design the bandpass filter.
# Note: The filter is designed for a sampling rate of 48000 Hz, which may differ from the recording RATE.
sos = design_filter(19400, 19600, 48000, 3)
                                # Q: Why might the filter use a different sampling rate?
                                # A: The filter parameters can be chosen based on a target application or hardware constraints.
                                
#%% Real-Time Audio Acquisition, Filtering, and Visualization
exec_time = []  # List to store execution times for the filtering operation.

# Calculate the total number of frames to record based on RATE, BUFFER size, and RECORD_SECONDS.
num_frames = (RATE // BUFFER) * RECORD_SECONDS

for _ in range(num_frames):
    # Record an audio frame from the microphone.
    # sd.rec() captures 'BUFFER' number of samples.
    data = sd.rec(frames=BUFFER, samplerate=RATE, channels=CHANNELS, dtype='int16', blocking=True)
    data = np.squeeze(data)  # Remove extra dimensions (if any) from the recorded data.
    
    # Apply bandpass filtering to the recorded frame.
    start_time = time.time()  # Start timer for filtering execution.
    yf = sosfilt(sos, data)    # Filter the data using the designed bandpass filter.
    exec_time.append(time.time() - start_time)  # Record the time taken for filtering.
    
    # Update the plot with the new raw and filtered data.
    line.set_ydata(data)       # Update raw audio waveform.
    line_filter.set_ydata(yf)    # Update filtered audio waveform.
    fig.canvas.draw()          # Redraw the canvas to show updated plots.
    fig.canvas.flush_events()  # Ensure the updated plots are rendered immediately.

#%% End of Streaming and Execution Time Reporting
print('Stream stopped')
print('Average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))
                                # Q: What does the average execution time indicate?
                                # A: It provides insight into the processing speed per frame, helping to evaluate the real-time performance.
