#!/usr/bin/env python3
"""
Top-Level Explanation:
This script captures live audio from the microphone, processes it in real-time, and visualizes both the time-domain waveform and its frequency-domain spectrum.
It uses PyAudio for audio capture, performs a Fast Fourier Transform (FFT) to analyze the audio frequencies, and updates plots using Matplotlib.
This type of code is useful for audio signal analysis, debugging audio hardware, or as a foundation for more advanced audio processing tasks.
"""

#%% Import the required libraries
import pyaudio  # For interfacing with the audio hardware (microphone input and audio output).
                # Q: Why use PyAudio? 
                # A: It provides a convenient interface to capture and play back audio data in Python.
import struct   # For converting raw audio bytes into numerical 16-bit integers.
                # Q: What is the purpose of the struct module?
                # A: It helps convert binary data (from the microphone) into a format that can be processed (integers).
import numpy as np  # For efficient numerical and array operations.
import matplotlib.pyplot as plt  # For plotting the audio waveform and spectrum.
from scipy.fftpack import fft, fftfreq  # For computing the FFT (transforms time-domain data to frequency-domain).
                # Q: What does FFT do?
                # A: It converts a signal from the time domain to the frequency domain, revealing its frequency components.
import time  # For measuring the execution time of operations (like the FFT computation).

#%% Parameters
# Define constants for audio processing and visualization
BUFFER = 1024 * 16           # Number of samples per frame. A larger buffer can provide higher frequency resolution.
                             # Q: Why is the buffer size important?
                             # A: It determines the resolution of both the time and frequency analysis.
FORMAT = pyaudio.paInt16     # Audio format, indicating that audio samples are 16-bit integers.
                             # Q: What does paInt16 indicate?
                             # A: It specifies that each audio sample is a 16-bit number.
CHANNELS = 1                 # Number of audio channels; 1 means mono audio.
                             # Q: What would be the effect of using 2 channels?
                             # A: It would capture stereo audio instead of mono.
RATE = 44100                 # Sampling rate in samples per second (44.1 kHz is standard for high-quality audio).
                             # Q: Why choose a sampling rate of 44100 Hz?
                             # A: It is a standard that provides a good balance between audio quality and processing requirements.
RECORD_SECONDS = 30          # Total duration (in seconds) to record audio from the microphone.

#%% Create matplotlib figure and axes with initial random plots as placeholders
# Set up a figure with two subplots: one for the waveform (time domain) and one for the spectrum (frequency domain).
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))

# Generate x-axis data for the waveform plot.
x = np.arange(0, 2 * BUFFER, 2)  # Sample indices for the waveform.
                # Q: Why is the x-axis generated with a step of 2?
                # A: It matches the length of the unpacked data array from the binary stream.
                
# Generate frequency bins for the FFT output.
xf = fftfreq(BUFFER, (1 / RATE))[:BUFFER // 2]
                # Q: What is fftfreq used for?
                # A: It computes the frequency values corresponding to the FFT bins.
                # Note: Only the first half of the FFT is used because the FFT of a real-valued signal is symmetric.

# Create initial line objects for the waveform and FFT plots using random data as placeholders.
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)
line_fft, = ax2.plot(xf, np.random.rand(BUFFER // 2), '-', lw=2)

# Format the waveform (time-domain) plot.
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('Samples')
ax1.set_ylabel('Amplitude (volume)')
ax1.set_ylim(-5000, 5000)  # Adjust y-axis limits to view the typical range of microphone signal amplitudes.
ax1.set_xlim(0, BUFFER)

# Format the frequency spectrum plot.
ax2.set_title('SPECTRUM')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude (Log Scale)')
ax2.set_ylim(0, 1000) 
ax2.set_xlim(0, RATE / 2)  # x-axis goes up to the Nyquist frequency (half the sampling rate).

# Display the plot in non-blocking mode so that the script continues to run.
plt.show(block=False)

#%% Initialize the pyaudio class instance
# Create a PyAudio object to interface with the audio hardware.
audio = pyaudio.PyAudio()

# Open an audio stream to capture data from the microphone.
stream = audio.open(
    format=FORMAT,           # Audio sample format.
    channels=CHANNELS,       # Number of channels (mono in this case).
    rate=RATE,               # Sampling rate.
    input=True,              # Enable input (recording).
    output=True,             # Enable output (if needed).
    frames_per_buffer=BUFFER # Number of samples per frame.
)
print('stream started')  # Indicate that audio capturing has begun.

#%% Main loop: Capture audio, compute FFT, and update the plots in real-time.
exec_time = []  # List to store the execution time of each FFT computation.

# Calculate the total number of frames to capture based on the recording duration and buffer size.
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):
    
    # Read a chunk of binary audio data from the microphone.
    data = stream.read(BUFFER)
    
    # Convert the binary data to a tuple of 16-bit integers.
    # The format string (e.g., '16384h' for a BUFFER of 16384 samples) tells struct.unpack how many integers to extract.
    data_int = struct.unpack(str(BUFFER) + 'h', data)
                # Q: Why do we need to convert binary data to integers?
                # A: To be able to perform numerical analysis (such as FFT) on the audio data.
    
    # Compute the FFT of the audio data.
    start_time = time.time()  # Start timing the FFT computation.
    yf = fft(data_int)        # Compute the Fast Fourier Transform.
    exec_time.append(time.time() - start_time)  # Record the time taken for the FFT.
    
    # Update the waveform plot with the new audio data.
    line.set_ydata(data_int)
    
    # Update the FFT plot:
    # Normalize the FFT data by scaling with 2.0/BUFFER and take the absolute value (magnitude).
    # Only the first half of the FFT is used since the second half is symmetric.
    line_fft.set_ydata(2.0 / BUFFER * np.abs(yf[0:BUFFER // 2]))
                # Q: Why normalize the FFT output?
                # A: Normalization scales the FFT data to reflect the actual amplitude of the frequency components.
    
    # Refresh the plots with the updated data.
    fig.canvas.draw()
    fig.canvas.flush_events()

#%% Cleanup: Terminate the audio stream and report execution time.
audio.terminate()  # Close the audio stream and release resources.
   
print('stream stopped')
# Print the average execution time for the FFT computation, in milliseconds.
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))
                # Q: Why measure the FFT execution time?
                # A: It helps determine if the processing is fast enough for real
