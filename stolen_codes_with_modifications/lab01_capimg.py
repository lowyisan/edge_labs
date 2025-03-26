#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script demonstrates a simple image capture functionality using OpenCV.
It initializes the webcam, captures a single frame, saves it as a JPEG image file,
and then releases the webcam resource. This example is useful for lab tests involving
basic video capture and file I/O operations in computer vision applications.

Potential Lab Test Questions:
Q1. What does cv2.VideoCapture(0) do?
   A1. It initializes video capturing from the default webcam (device index 0).
Q2. How is the success of the frame capture verified in the code?
   A2. The boolean variable 'ret' returned by cap.read() indicates if the frame was successfully captured.
Q3. Why is it important to release the webcam resource after capturing the frame?
   A3. To free up system resources and avoid potential conflicts with other applications.
"""

# %% Import the necessary library
import cv2  # OpenCV library for image capture and processing

# %% Initialize the webcam
cap = cv2.VideoCapture(0)  # Opens the default webcam (device index 0)
# Q: Why do we use device index 0?
# A: Device index 0 typically refers to the default camera on the system.

# %% Check if the webcam is successfully opened
if not cap.isOpened():
    # If the webcam fails to open, raise an error to alert the user.
    raise IOError("Cannot open webcam")
# Q: What does cap.isOpened() check for?
# A: It verifies that the webcam has been successfully initialized for video capture.

# %% Capture a single frame from the webcam
ret, frame = cap.read()  # 'ret' is a boolean indicating success; 'frame' is the captured image.
# Q: How do we know if the frame was captured successfully?
# A: The boolean 'ret' will be True if the capture was successful.

# %% Save the captured frame as an image file
cv2.imwrite('captured_image.jpg', frame)  # Saves the captured frame to a JPEG file.
# Q: What function is used to write the image to disk?
# A: cv2.imwrite() is used to save the image file.

# %% Release the webcam resource
cap.release()  # Releases the webcam, freeing the hardware for other applications.
# Q: Why is it important to call cap.release()?
# A: It ensures that system resources are freed and prevents conflicts with other programs.

# %% Print a confirmation message
print("Image captured and saved!")
# Q: What feedback does the program provide to the user?
# A: It prints a confirmation message indicating that the image has been successfully captured and saved.
