#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script implements an audio analysis pipeline using Librosa for feature extraction
and Matplotlib for visualization. It loads an audio file, computes and plots several
representations of the audio data including the waveform, spectrogram, chromagram, 
Mel-spectrogram, and MFCCs. These features are crucial in tasks such as speech recognition,
music genre classification, and audio fingerprinting.

Potential Lab Test Questions:
Q1. What is the purpose of using a spectrogram in audio analysis?
   A1. A spectrogram provides a time-frequency representation of an audio signal,
       which is useful for visualizing how the frequency content of the audio changes over time.
Q2. Why are MFCCs important in speech recognition?
   A2. MFCCs capture the short-term spectral shape of an audio signal, which closely
       correlates with how humans perceive sound, making them valuable features for speech recognition.
Q3. How does the Mel scale improve the representation of the audio signal?
   A3. The Mel scale compresses the frequency axis to reflect human auditory perception,
       emphasizing frequencies that humans are more sensitive to.
"""

#%% Import the required libraries
import numpy as np                      # For numerical operations and array handling
import matplotlib.pyplot as plt         # For plotting and visualizations
import librosa                          # Core library for audio loading and feature extraction
import librosa.display                  # For advanced audio-specific plotting

#%% Load an audio file
# Load audio from 'test.wav'. Setting sr=None preserves the original sample rate.
y, sr = librosa.load("test.wav", sr=None)
# y: audio time series as a 1D NumPy array
# sr: sampling rate of the loaded audio
# Q: Why set sr=None when loading the audio?
# A: To keep the original sample rate of the file without any resampling.

#%% Compute the Spectrogram
# Compute the complex Short-Time Fourier Transform (STFT) of the audio signal.
# The function returns a complex matrix; librosa.magphase separates the magnitude and phase.
S_full, phase = librosa.magphase(librosa.stft(y))
# S_full: magnitude spectrogram (used for visualization)
# phase: phase component (not used further in this script)
# Q: What does the STFT provide in audio analysis?
# A: It converts the audio signal into a time-frequency domain, enabling visualization of how frequency components vary over time.

#%% Plot Waveform and Spectrogram (Time-Frequency Representation)
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))

# === Time-domain waveform plot ===
ax1.plot(y)
ax1.set_xlabel('Samples')
ax1.set_ylabel('Amplitude')
ax1.set(title='Time Series')
# Q: Why visualize the waveform?
# A: To understand the amplitude variations of the audio signal over time.

# === Spectrogram plot ===
# Convert amplitude values to decibels for better visualization and use a logarithmic frequency scale.
img = librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                               y_axis='log', x_axis='time', sr=sr, ax=ax2)
fig.colorbar(img, ax=ax2)
ax2.set(title='Spectrogram')
# Q: What advantage does displaying the spectrogram in dB scale offer?
# A: It provides a more intuitive visualization of the dynamic range of the signal.

# Display both plots together
plt.show()

#%% Chroma Estimation (Pitch Class Energy over Time)
# Chroma features represent the energy distribution among the 12 pitch classes (e.g., C, C#, D, ... B)
# which is useful for analyzing harmonic and chordal aspects of the audio.

# Compute power spectrogram using a larger FFT window for improved frequency resolution.
S = np.abs(librosa.stft(y, n_fft=4096))**2

# Extract chromagram from the power spectrogram.
chroma = librosa.feature.chroma_stft(S=S, sr=sr)
# Q: What information does a chromagram provide?
# A: It shows the intensity of each of the 12 pitch classes over time, which is helpful for tasks like chord detection and key estimation.

# Plot the power spectrogram and the chromagram.
fig, ax = plt.subplots(nrows=2, sharex=True)

# === Power spectrogram ===
img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].label_outer()
ax[0].set(title='Power Spectrogram')

# === Chroma plot ===
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='Chromagram')

plt.show()

#%% Compute Mel-Spectrogram
# The Mel spectrogram compresses the frequency axis to match human auditory perception.
# This representation is widely used in audio classification tasks.

S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
# Convert the Mel spectrogram to decibel (dB) units for visualization.
S_mel_dB = librosa.power_to_db(S_mel, ref=np.max)

# Plot the Mel spectrogram.
fig, ax = plt.subplots()
img = librosa.display.specshow(S_mel_dB, x_axis='time',
                               y_axis='mel', sr=sr,
                               fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency Spectrogram')
plt.show()

#%% Compute MFCC (Mel-Frequency Cepstral Coefficients)
# MFCCs capture the short-term power spectrum of the audio and are key features in speech recognition.

# Compute 40 MFCCs from the audio signal.
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

# Also compute the Mel spectrogram for reference in the same plot.
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Plot both the Mel spectrogram and the MFCCs.
fig, ax = plt.subplots(nrows=2, sharex=True)

# === Mel spectrogram ===
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel Spectrogram')
ax[0].label_outer()

# === MFCC plot ===
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')

plt.show()

#%% Additional Use Cases and Extensions
'''
Use Cases:
- Speech recognition and emotion detection.
- Music genre classification.
- Audio fingerprinting and retrieval.
- Preprocessing for deep learning audio models.

Possible Additions:
- Implement pitch estimation using functions like librosa.yin().
- Estimate tempo using librosa.beat.tempo().
- Save computed features (e.g., MFCCs, chroma) as .npy or .csv files for training ML models.
- Segment the audio features using windowing for real-time classification tasks.
'''
