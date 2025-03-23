#!/usr/bin/env python3
"""
Top-Level Explanation:
This script is a demonstration of speech recognition using the SpeechRecognition library.
It captures audio input from the microphone, processes the audio to recognize spoken words,
and then prints the recognized text using two different recognition engines:
Google Speech Recognition (an online service) and CMU Sphinx (an offline engine).
This demo is useful for learning how to integrate speech recognition into Python projects,
handle ambient noise, and manage potential errors during the recognition process.
Refer to the library's GitHub page for more details:
https://github.com/Uberi/speech_recognition?tab=readme-ov-file#readme
"""

#%% Import all necessary libraries
import speech_recognition as sr  # Library to access various speech recognition engines.
                                # Q: Why use the SpeechRecognition library?
                                # A: It provides a simple interface for capturing audio and converting speech to text.
import time                    # Used for timing operations, such as measuring recognition duration.
import os                      # Used to execute operating system commands (e.g., clearing the terminal).

#%% Recording from microphone
# Obtain audio input from the microphone.

# Initialize the Recognizer instance; this object handles the audio processing.
r = sr.Recognizer()  # Q: What is the purpose of the Recognizer class?
                     # A: It encapsulates the functionality to capture and process audio for speech recognition.

# Use the microphone as the source for audio input.
with sr.Microphone() as source:
    # Adjust the recognizer sensitivity to ambient noise.
    # This step calibrates the recognizer to the background noise level.
    r.adjust_for_ambient_noise(source)
    # Q: Why is adjusting for ambient noise important?
    # A: It helps reduce the impact of background sounds, making the speech recognition more accurate.
    
    # Clear the terminal screen for a cleaner output.
    os.system('clear')
    print("Say something!")
    
    # Listen for the first phrase and capture it as audio.
    audio = r.listen(source)
    # Q: What does r.listen(source) do?
    # A: It records audio from the microphone until a pause is detected.

#%% Recognize speech using Google Speech Recognition
# Measure the time taken for Google Speech Recognition to process the audio.
start_time = time.time()  # Start timing the recognition process.

try:
    # Recognize the captured audio using Google's speech recognition service.
    # The default API key is used for testing purposes.
    recognized_text = r.recognize_google(audio)
    print("Google Speech Recognition thinks you said " + recognized_text)
    # Q: How does r.recognize_google(audio) work?
    # A: It sends the audio data to Google's online service and returns the recognized text.
except sr.UnknownValueError:
    # This error is raised when the audio is not understood.
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    # This error is raised when there is a problem with the request (e.g., network issues).
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Print the time taken for Google Speech Recognition.
print('Time for Google Speech Recognition recognition = {:.0f} seconds'.format(time.time() - start_time))

#%% Recognize speech using Sphinx
# Measure the time taken for CMU Sphinx (an offline engine) to process the audio.
start_time = time.time()  # Restart timing for Sphinx recognition.

try:
    # Recognize the captured audio using CMU Sphinx.
    recognized_text = r.recognize_sphinx(audio)
    print("Sphinx thinks you said " + recognized_text)
    # Q: What is the advantage of using Sphinx over Google Speech Recognition?
    # A: Sphinx is an offline engine, which means it doesn't require an internet connection.
except sr.UnknownValueError:
    # This error is raised when Sphinx cannot understand the audio.
    print("Sphinx could not understand audio")
except sr.RequestError as e:
    # This error is raised if there is an issue with Sphinx (e.g., missing data files).
    print("Sphinx error; {0}".format(e))

# Print the time taken for Sphinx recognition.
print('Time for Sphinx recognition = {:.0f} seconds'.format(time.time() - start_time))
