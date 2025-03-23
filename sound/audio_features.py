#!/usr/bin/env python3
"""
Top-Level Explanation:
This script demonstrates various audio feature extraction techniques using the Librosa library.
It loads an audio file ("test.wav"), computes its short-time Fourier transform (STFT) to generate a spectrogram,
estimates chroma features, computes a Mel-spectrogram, and extracts MFCC (Mel-frequency cepstral coefficients).
Each of these features is visualized using Matplotlib.
This code is useful for understanding how to extract and analyze different audio features, which is a common task
in speech and audio processing, and could be helpful for your lab test.
"""

#%% Import the required libraries
import numpy as np                      # Provides efficient numerical operations on arrays.
import matplotlib.pyplot as plt         # For plotting graphs and visualizations.
import librosa                         # Library for audio processing and feature extraction.
                                        # Q: Why use Librosa?
                                        # A: It offers powerful tools for audio analysis, including loading audio, computing spectrograms, and extracting features.
# Note: librosa.display is used for visualizing spectrograms; make sure it is available in your Librosa version.
import librosa.display

#%% Load Audio File
# Load the audio file "test.wav" with its original sampling rate (sr=None preserves the file's native rate).
y, sr = librosa.load("test.wav", sr=None)
# Q: What do 'y' and 'sr' represent?
# A: 'y' is the audio time series (a NumPy array of amplitude values), and 'sr' is the sampling rate (samples per second).

#%% Compute the Spectrogram Magnitude and Phase
# Compute the short-time Fourier transform (STFT) of the audio signal.
# Then, separate the magnitude and phase components.
S_full, phase = librosa.magphase(librosa.stft(y))
# Q: What is the purpose of computing the STFT?
# A: STFT converts the time-domain signal into the frequency domain, allowing analysis of how frequency content evolves over time.

#%% Plot the Time Series and the Frequency-Time Plot (Spectrogram)
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))
# Plot the time series (raw audio signal).
ax1.plot(y)
ax1.set_xlabel('Samples')
ax1.set_ylabel('Volume')
ax1.set(title='Time Series')
# Q: Why visualize the time series?
# A: It shows the raw amplitude variation over time and helps in understanding the overall dynamics of the audio.

# Convert the amplitude spectrogram to decibel (dB) units for better visualization.
img = librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                               y_axis='log', x_axis='time', sr=sr, ax=ax2)
fig.colorbar(img, ax=ax2)
ax2.set(title='Spectrogram')
plt.show()

#%% Chroma Estimation
# Compute the power spectrogram (magnitude squared) from the STFT.
S = np.abs(librosa.stft(y, n_fft=4096))**2
# Extract chroma features from the power spectrogram.
chroma = librosa.feature.chroma_stft(S=S, sr=sr)
# Q: What are chroma features?
# A: Chroma features represent the intensity of each of the 12 different pitch classes (semitones) in the audio.

# Plot the power spectrogram and the chromagram.
fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Power Spectrogram')
ax[0].label_outer()

img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='Chromagram')
plt.show()

#%% Compute Mel-Spectrogram
# Compute a Mel-scaled spectrogram with 128 Mel bands and a maximum frequency of 8000 Hz.
S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
# Convert the Mel spectrogram power to decibel (dB) units.
S_mel_dB = librosa.power_to_db(S_mel, ref=np.max)
# Plot the Mel-frequency spectrogram.
fig, ax = plt.subplots()
img = librosa.display.specshow(S_mel_dB, x_axis='time', y_axis='mel',
                               sr=sr, fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-Frequency Spectrogram')
plt.show()

#%% Compute MFCC (Mel-Frequency Cepstral Coefficients)
# Extract 40 MFCCs from the audio signal.
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
# Recompute the Mel spectrogram for visualization.
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
# Plot both the Mel spectrogram and the MFCCs.
fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000, ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel Spectrogram')
ax[0].label_outer()

img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')
plt.show()
# Q: What are MFCCs?
# A: MFCCs (Mel-frequency cepstral coefficients) are features that capture the timbral aspects of audio, commonly used in speech and music analysis.
