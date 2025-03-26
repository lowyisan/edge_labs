#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script demonstrates a basic speech recognition pipeline using the SpeechRecognition library.
It captures audio from the default microphone, adjusts for ambient noise, and then attempts to convert the
spoken words into text using two different engines: Google Web Speech API (online) and PocketSphinx (offline).
This example is useful for lab tests covering speech-to-text systems, API integration, and handling both online
and offline recognition scenarios.

Potential Lab Test Questions:
Q1. What is the purpose of using r.adjust_for_ambient_noise(source)?
   A1. It calibrates the recognizer to ignore ambient noise so that the speech recognition is more accurate.
Q2. How does the Google Web Speech API differ from PocketSphinx in this script?
   A2. The Google Web Speech API is an online service providing high accuracy but requires internet connectivity,
       while PocketSphinx works offline but may have lower accuracy.
Q3. What exceptions are handled in this script, and why are they important?
   A3. The script handles UnknownValueError (when speech is unintelligible) and RequestError (when the API cannot be reached),
       ensuring the program gracefully reports issues during recognition.
"""

#%% Import required libraries

import speech_recognition as sr  # Core library for converting speech to text
# See: https://github.com/Uberi/speech_recognition

import time  # For measuring the time taken for recognition
import os    # For clearing the terminal (optional)

#%% Recording from Microphone

# Create a Recognizer instance for speech recognition operations
r = sr.Recognizer()

# Use the default microphone as the audio source
with sr.Microphone() as source:
    # Adjust the recognizer sensitivity to ambient noise to improve accuracy.
    # This requires a short period of silence (1-2 seconds) to calibrate.
    r.adjust_for_ambient_noise(source)

    # Clear the terminal for a clean interface (works on Unix-based systems)
    os.system('clear')

    print("Say something!")  # Prompt the user to speak

    # Listen to the source and capture the audio data.
    # This call blocks until a phrase is detected and then stops automatically.
    audio = r.listen(source)

#%% Recognize Speech using Google Web API (Online)

start_time = time.time()  # Start timer to measure recognition time

try:
    # Attempt to recognize the speech using Google's Web Speech API.
    # Note: This online service requires an active internet connection.
    # Optionally, you can provide a Google API key using the key parameter.
    text = r.recognize_google(audio)
    print("Google Speech Recognition thinks you said: " + text)

except sr.UnknownValueError:
    # Raised when the speech is unintelligible
    print("Google Speech Recognition could not understand audio")

except sr.RequestError as e:
    # Raised when the request to the API fails (e.g., no internet connection)
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Print how long the Google recognition took
print('Time for Google Speech Recognition = {:.0f} seconds'.format(time.time() - start_time))

#%% Recognize Speech using PocketSphinx (Offline)

start_time = time.time()  # Restart timer for offline recognition

try:
    # Attempt to recognize the speech using PocketSphinx (offline).
    # Ensure that PocketSphinx is installed via: pip install pocketsphinx
    text_sphinx = r.recognize_sphinx(audio)
    print("Sphinx thinks you said: " + text_sphinx)

except sr.UnknownValueError:
    # Raised when PocketSphinx cannot interpret the spoken words
    print("Sphinx could not understand audio")

except sr.RequestError as e:
    # Raised if there's an internal error in the Sphinx recognition process
    print("Sphinx error: {0}".format(e))

# Print how long the PocketSphinx recognition took
print('Time for Sphinx recognition = {:.0f} seconds'.format(time.time() - start_time))

# =================================================================
'''
    Metrics         Google API      vs.      PocketSphinx
    ---------------------------------------------------------
    Internet         Required                Not Required (Offline)
    Accuracy         High                    Lower
    Speed            Variable (depends on connection)  Fast

- Google API is suitable for production demos when internet is available.
- PocketSphinx is a good fallback for offline applications or testing.
'''
