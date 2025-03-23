#!/usr/bin/env python3
"""
Top-Level Explanation:
This script uses OpenCV's Haar Cascade classifier to perform real-time face detection from a webcam feed.
It captures video frames, converts them to grayscale, and applies a pre-trained Haar Cascade model to detect faces.
Detected faces are highlighted by drawing white rectangles around them. This demo is useful for understanding
object detection, the use of pre-trained models (Haar Cascades), and real-time video processing in computer vision.
Before running the code, ensure that the Haar Cascade XML file (haarcascade_frontalface_alt2.xml) is in the same folder.
Download it from:
https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml
"""

import cv2  # OpenCV for image and video processing.
           # Q: Why use OpenCV for face detection?
           # A: OpenCV provides efficient and easy-to-use computer vision tools, including pre-trained Haar Cascade models.

#%% Initiate the Face Detection Cascade Classifier
haarcascade = "haarcascade_frontalface_alt2.xml"  # Path to the Haar Cascade model file.
detector = cv2.CascadeClassifier(haarcascade)      # Load the Haar Cascade classifier for face detection.
           # Q: What is a Haar Cascade?
           # A: It is a machine learning-based approach for detecting objects, like faces, based on Haar features.

#%% OpenCV Video Capture and Frame Analysis
cap = cv2.VideoCapture(0)  # Open the default webcam (index 0).
           # Q: What does cv2.VideoCapture(0) do?
           # A: It initializes the video capture from the system's default camera.

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")  # Raise an error if the webcam cannot be accessed.

# Process video frames continuously until 'q' is pressed
while True:
    try:
        # Capture one frame from the webcam.
        ret, frame = cap.read()
        # Q: What do ret and frame represent?
        # A: 'ret' is a boolean indicating if the frame was successfully captured; 'frame' is the captured image.

        # Resize the frame to 256x256 pixels for faster processing.
        frame = cv2.resize(frame, (256, 256))
        # Q: How does resizing help?
        # A: Smaller frames require less computation, increasing the speed of detection.

        # Convert the frame to grayscale, since Haar Cascade works on grayscale images.
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Q: Why convert to grayscale?
        # A: Feature extraction with Haar Cascades is based on intensity differences, which are more efficient on single-channel images.

        # Detect faces in the grayscale image using the Haar Cascade classifier.
        faces = detector.detectMultiScale(image_gray)
        # Q: What does detectMultiScale() return?
        # A: It returns a list of bounding boxes around detected faces in the format (x, y, width, height).

        # Loop over the detected faces and draw rectangles around them.
        for face in faces:
            (x, y, w, d) = face
            # Draw a white rectangle around the face on the original frame.
            cv2.rectangle(frame, (x, y), (x + w, y + d), (255, 255, 255), 2)
            # Q: What do the parameters of cv2.rectangle() represent?
            # A: They represent the image, top-left and bottom-right corners of the rectangle, the color (BGR), and the thickness.

        # Resize the frame for display (making it larger for easier viewing).
        frame = cv2.resize(frame, (720, 720))
        # Display the processed frame in a window titled "frame".
        cv2.imshow("frame", frame)

        # Exit the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Allow graceful exit on keyboard interrupt (e.g., Ctrl+C).
        break

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
# Q: Why is it important to release the capture and destroy windows?
# A: To free system resources and ensure that the program terminates cleanly.
