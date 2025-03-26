#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script implements a basic motion detection system using OpenCV. It captures video from the default webcam,
computes the difference between consecutive frames to detect changes (indicative of motion), processes the
resulting difference image through grayscale conversion, blurring, thresholding, and dilation, and then finds
and highlights contours (areas of motion) by drawing bounding rectangles on the frame. This approach demonstrates
how to perform real-time image processing and motion detection, which is useful for lab tests in computer vision
and surveillance applications.

Potential Lab Test Questions:
Q1. How does frame differencing help in detecting motion?
   A1. Frame differencing highlights areas of change between consecutive frames, which can indicate motion.
Q2. Why are operations like blurring and thresholding used in this script?
   A2. They reduce noise and emphasize significant changes, improving the accuracy of motion detection.
Q3. What is the purpose of contour detection in this context?
   A3. Contour detection is used to identify and bound regions where motion has occurred.
"""

# %% Import necessary libraries
import cv2  # OpenCV library for image processing and video capture
import numpy as np  # Used for array operations and numerical computations (common in image processing)

# %% Initialize the webcam
cap = cv2.VideoCapture(0)  # Open the default webcam (device index 0)
# Q: What does cv2.VideoCapture(0) do?
# A: It initializes video capturing from the default webcam.

# %% Capture initial frames for motion comparison
# Read the first two frames from the webcam
_, frame1 = cap.read()  # First frame capture; the underscore ignores the success flag for brevity
_, frame2 = cap.read()  # Second frame capture for initial comparison

# %% Main processing loop: Continues until 'q' is pressed
while True:
    # === MOTION DETECTION PREPROCESSING ===

    # Compute the absolute difference between the two frames
    diff = cv2.absdiff(frame1, frame2)
    # Q: What does cv2.absdiff() do?
    # A: It calculates the absolute difference between two images, highlighting the pixel changes that may represent motion.

    # Convert the diff image to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Q: Why convert to grayscale?
    # A: Grayscale simplifies processing by reducing the image to one color channel, making further processing faster and less complex.

    # Blur the grayscale image to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Q: What is the purpose of applying Gaussian blur?
    # A: Blurring reduces noise and smooths the image, which helps to prevent false detections during thresholding.

    # Apply binary thresholding to isolate significant differences (i.e., areas of motion)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # Q: What does thresholding achieve here?
    # A: It converts the image into a binary form, where pixels with intensity above the threshold become white (255)
    #    and those below become black (0), emphasizing regions with significant change.

    # Dilate the thresholded image to fill in holes and connect regions
    dilated = cv2.dilate(thresh, None, iterations=3)
    # Q: Why use dilation?
    # A: Dilation expands the white regions, helping to fill small holes and connect disjoint parts of the moving object.

    # Find contours in the dilated image to identify regions of motion
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Q: What is the role of cv2.findContours()?
    # A: It retrieves the boundaries of objects (contours) from the binary image, which can then be used to locate moving regions.

    # === DRAW DETECTIONS ON FRAME ===
    for contour in contours:
        # Get bounding box coordinates for the current contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Ignore contours with small areas to filter out noise and minor changes
        if cv2.contourArea(contour) < 900:
            continue

        # Draw a green rectangle around the detected motion on the first frame
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Overlay status text indicating movement on the frame
        cv2.putText(frame1, "Status: Movement", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display the frame with the drawn rectangles and text in a window
    cv2.imshow("feed", frame1)

    # Shift frames: prepare for the next iteration by setting frame1 to frame2 and reading a new frame into frame2
    frame1 = frame2
    _, frame2 = cap.read()  # Capture a new frame for future comparison

    # Check for key press: exit the loop if the 'q' key is pressed
    if cv2.waitKey(40) == ord('q'):
        break

# %% Cleanup: Release resources and close windows after the loop ends
cap.release()            # Release the webcam resource
cv2.destroyAllWindows()  # Close all OpenCV windows
# Q: Why is cleanup necessary?
# A: Releasing resources and closing windows helps prevent memory leaks and frees system resources after the program ends.

# ============== ADDITIONAL EXPLANATIONS AND QUESTIONS ==================
'''
Q1. Identify and explain the additional functionalities introduced in this motion detection code.
   - The code captures continuous video frames, computes differences between consecutive frames to detect motion,
     applies pre-processing techniques (grayscale conversion, blurring, thresholding, and dilation) to filter noise,
     and uses contour detection to locate and highlight moving objects by drawing rectangles.
   - This transforms a simple image capture into a real-time movement detection system.

Q2. How do the OpenCV functions cv2.absdiff, cv2.cvtColor, cv2.GaussianBlur, cv2.threshold, cv2.dilate, and cv2.findContours
    contribute to motion detection?
   - cv2.absdiff: Highlights changes between frames.
   - cv2.cvtColor: Converts images to grayscale for simpler processing.
   - cv2.GaussianBlur: Smooths the image to reduce noise.
   - cv2.threshold: Binarizes the image, isolating significant changes.
   - cv2.dilate: Expands white areas to connect regions.
   - cv2.findContours: Detects the boundaries of moving objects.

Q3. Why is there a condition to ignore contours with areas less than 900?
   - This condition filters out small contours that likely represent noise or minor movements, ensuring that only significant motion is detected.

Q4. Explain the role of the while loop in this script.
   - The while loop continuously captures and processes frames in real-time, enabling constant monitoring for motion detection.
   - This is in contrast to a single capture approach, which would only provide a static snapshot.

Q5. What are some possible improvements or additional features that could be implemented in this motion detection system?
   - Implement background subtraction models for more robust motion detection.
   - Add functionality to record video or capture snapshots only when motion is detected.
   - Integrate a timestamp overlay for logging the time of detected motion.
   - Use morphological operations like erosion before dilation to further reduce noise.
'''
