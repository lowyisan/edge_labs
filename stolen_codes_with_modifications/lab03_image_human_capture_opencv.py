#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script demonstrates face detection using OpenCV's Haar Cascade classifier. It captures video from the webcam,
converts each frame to grayscale (as Haar Cascade works best with grayscale images), and applies the face detector
to locate faces. Detected faces are highlighted with a white rectangle. The script also demonstrates how to handle
optional resizing for performance and display purposes. This example is useful for lab tests covering real-time
face detection, image preprocessing, and object detection with machine learning models.

Potential Lab Test Questions:
Q1. Why do we convert the captured frame to grayscale before running the face detector?
   A1. Haar Cascade classifiers are trained on grayscale images, and converting to grayscale reduces computational complexity.
Q2. What is the role of the detectMultiScale() function?
   A2. It detects objects (faces) in the image and returns bounding boxes around each detected face.
Q3. How can adjusting the parameters of detectMultiScale() affect detection performance?
   A3. Parameters such as scaleFactor, minNeighbors, and minSize control detection sensitivity and speed; fine-tuning these can help reduce false positives and improve detection speed.
"""

#%% Import required libraries
import cv2  # OpenCV library for computer vision and webcam processing

#%% Initialize Haar Cascade Face Detector

# Path to the Haar Cascade classifier file for detecting frontal faces.
# Ensure that "haarcascade_frontalface_alt2.xml" is in the same directory as this script.
haarcascade = "haarcascade_frontalface_alt2.xml"

# Load the Haar Cascade model from the XML file.
detector = cv2.CascadeClassifier(haarcascade)

#%% Start Webcam Video Capture

# Open the default webcam (device index 0)
cap = cv2.VideoCapture(0)

# Validate the webcam connection; exit if the webcam cannot be accessed.
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#%% Main Frame Processing Loop

# This loop continues until 'q' is pressed or a KeyboardInterrupt occurs.
while True:
    try:
        # Capture a single frame from the webcam.
        ret, frame = cap.read()

        # OPTIONAL: Resize the frame to 256x256 for faster processing.
        # This can improve detection speed on lower performance systems.
        frame = cv2.resize(frame, (256, 256))  # Comment out to use the original frame size.

        # Convert the color image (BGR) to grayscale.
        # Haar Cascades work best on grayscale images for simplicity and efficiency.
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run the face detector on the grayscale image.
        # detectMultiScale returns a list of bounding boxes (x, y, width, height) for each detected face.
        faces = detector.detectMultiScale(image_gray)

        # Loop through each detected face and draw a rectangle.
        for face in faces:
            (x, y, w, d) = face  # x, y are the top-left coordinates; w and d are the width and height.

            # Draw a white rectangle around the detected face on the original frame.
            # Parameters: top-left corner, bottom-right corner, color (white), thickness.
            cv2.rectangle(frame, (x, y), (x + w, y + d), (255, 255, 255), 2)

        # OPTIONAL: Resize the frame to a larger size (720x720 pixels) for display.
        frame = cv2.resize(frame, (720, 720))

        # Display the processed frame in a window titled "frame".
        cv2.imshow("frame", frame)

        # Exit the loop when the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Allow graceful exit using Ctrl+C.
        break

#%% Cleanup After Exit

# Release the webcam resource.
cap.release()

# Close any OpenCV GUI windows.
cv2.destroyAllWindows()

'''
detectMultiScale() Parameters Example:

faces = detector.detectMultiScale(
    image_gray,
    scaleFactor=1.1,       # How much the image size is reduced at each scale.
    minNeighbors=5,        # How many neighbors each candidate rectangle should have to retain it.
    minSize=(30, 30),      # Minimum possible object size to detect.
    flags=cv2.CASCADE_SCALE_IMAGE
)
'''
