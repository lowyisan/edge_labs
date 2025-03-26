#!/usr/bin/env python3
"""
Top-Level Explanation:
This script captures live audio from a microphone, applies a bandpass filter to the audio signal,
and displays both the original and filtered waveforms in real-time using Matplotlib.
It uses PyAudio for audio capture, SciPy’s signal processing functions (butter and sosfilt)
for designing and applying a bandpass filter, and NumPy for numerical operations.
This demo is useful for understanding real-time audio processing, filtering, and visualization,
and it can help you prepare for lab questions on digital signal processing and Python audio handling.
"""

#%% Import the required libraries
import pyaudio  # Interfaces with audio hardware for capturing microphone input.
              # Q: Why use PyAudio? 
              # A: It allows easy real-time audio capture and playback in Python.
import struct   # Used to convert binary audio data (bytes) into 16-bit integers.
              # Q: What is the role of the struct module?
              # A: It unpacks the raw byte data from the microphone into numerical values.
import numpy as np  # Provides efficient numerical operations and array manipulation.
import matplotlib.pyplot as plt  # For plotting waveforms and updating plots in real-time.
from scipy.signal import butter, sosfilt  # Used for designing and applying a bandpass filter.
              # Q: What do butter and sosfilt do?
              # A: 'butter' designs a Butterworth filter, and 'sosfilt' applies the filter in second-order sections.
import time  # For measuring execution time of operations (e.g., filtering frame rate).

#%% Parameters
# Set the parameters for audio capture and processing.
BUFFER = 1024 * 16          # Number of audio samples per frame.
                           # Q: Why is the buffer size important?
                           # A: It affects the resolution and latency of the real-time processing.
FORMAT = pyaudio.paInt16    # Audio format: 16-bit integer samples.
                           # Q: What does paInt16 represent?
                           # A: Each audio sample is stored as a 16-bit integer.
CHANNELS = 1                # Use a single channel (mono audio).
                           # Q: How would this change for stereo?
                           # A: CHANNELS would be set to 2.
RATE = 44100                # Sampling rate in samples per second (standard high-quality audio).
                           # Q: Why choose 44100 Hz?
                           # A: It's the CD-quality sampling rate providing a good balance between quality and processing load.
RECORD_SECONDS = 20         # Duration (in seconds) for which audio is recorded.

#%% Create matplotlib figure and axes with initial placeholder plots
# Create a figure with two subplots: one for the original audio waveform and one for the filtered output.
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))

# Generate x-axis data for the waveform plots.
x = np.arange(0, 2 * BUFFER, 2)  # Sample indices for the waveform.
                                # Q: Why use np.arange with step=2?
                                # A: It corresponds to the number of 16-bit samples after unpacking the byte stream.

# Create initial line objects with random data for both plots.
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)         # Original waveform plot.
line_filter, = ax2.plot(x, np.random.rand(BUFFER), '-', lw=2)    # Filtered waveform plot.

# Basic formatting for the original waveform plot.
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('Samples')
ax1.set_ylabel('Amplitude')
ax1.set_ylim(-5000, 5000)  # Adjust y-axis limits based on expected microphone signal amplitude.
ax1.set_xlim(0, BUFFER)

# Basic formatting for the filtered waveform plot.
ax2.set_title('FILTERED')
ax2.set_xlabel('Samples')
ax2.set_ylabel('Amplitude')
ax2.set_ylim(-5000, 5000)
ax2.set_xlim(0, BUFFER)

# Display the plot in non-blocking mode so the script continues execution.
plt.show(block=False)

#%% Function for design of filter
def design_filter(lowfreq, highfreq, fs, order=3):
    """
    Designs a Butterworth bandpass filter.
    
    Parameters:
        lowfreq (float): Lower cutoff frequency in Hz.
        highfreq (float): Higher cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): The order of the filter (default is 3).
    
    Returns:
        sos (ndarray): Second-order sections representation of the filter.
        
    Q: Why use second-order sections (sos)?
    A: They provide improved numerical stability for high-order filters.
    """
    nyq = 0.5 * fs  # Calculate the Nyquist frequency.
    low = lowfreq / nyq  # Normalize the low cutoff frequency.
    high = highfreq / nyq  # Normalize the high cutoff frequency.
    sos = butter(order, [low, high], btype='band', output='sos')  # Design the bandpass filter.
    return sos

# Design the filter with chosen cutoff frequencies.
# Note: Adjust the cutoff frequencies as needed. Here, it's set between 19400 Hz and 19600 Hz.
sos = design_filter(19400, 19600, 48000, 3)
              # Q: Why is the sampling frequency set to 48000 here?
              # A: This value is used for filter design; ensure it matches or is appropriately related to your audio RATE.

#%% Initialize the pyaudio class instance
# Create a PyAudio object to interact with the audio hardware.
audio = pyaudio.PyAudio()

# Open an audio stream to capture data from the microphone.
stream = audio.open(
    format=FORMAT,           # Audio format (16-bit integer).
    channels=CHANNELS,       # Mono audio.
    rate=RATE,               # Sampling rate.
    input=True,              # Enable audio input.
    output=True,             # Enable audio output (if needed).
    frames_per_buffer=BUFFER # Number of samples per frame.
)

print('stream started')  # Indicate that the audio stream has started.

#%% Main loop: Capture audio, apply filter, and update plots in real-time.
exec_time = []  # List to store the execution time for each filtering operation.

# Loop for the total duration calculated by dividing total samples (RATE * RECORD_SECONDS) by BUFFER.
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):
    
    # Read a chunk of binary audio data from the microphone.
    data = stream.read(BUFFER)
    
    # Convert the binary data to a tuple of 16-bit integers.
    # The format string (e.g., '16384h' for BUFFER = 16384) tells struct.unpack the number of samples.
    data_int = struct.unpack(str(BUFFER) + 'h', data)
                # Q: Why convert the byte data to integers?
                # A: It allows numerical processing (such as filtering and plotting) of the audio signal.
    
    # Apply bandpass filtering to the audio data.
    start_time = time.time()  # Start timing the filtering operation.
    yf = sosfilt(sos, data_int)  # Apply the designed bandpass filter.
    exec_time.append(time.time() - start_time)  # Record the time taken for filtering.
    
    # Update the original waveform plot with the new audio data.
    line.set_ydata(data_int)
    
    # Update the filtered waveform plot with the filtered data.
    line_filter.set_ydata(yf)
    
    # Redraw the updated plots.
    fig.canvas.draw()
    fig.canvas.flush_events()

#%% Cleanup: Terminate the audio stream and report the average execution time.
audio.terminate()  # Close the audio stream and free resources.

print('stream stopped')
# Calculate and print the average execution time for the filtering operation in milliseconds.
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))
                # Q: Why is measuring execution time important?
                # A: It helps determine if the filtering and plotting can be done in real-time.
