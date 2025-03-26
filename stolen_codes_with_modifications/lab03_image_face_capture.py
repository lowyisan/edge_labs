#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script utilizes MediaPipe and OpenCV to perform real-time face mesh detection using the webcam.
It initializes MediaPipe’s FaceMesh module to detect facial landmarks and draws the mesh tessellation
and contours over the detected face. This demonstration is useful for lab tests on computer vision,
real-time facial landmark detection, and augmented reality applications.

Potential Lab Test Questions:
Q1. Why do we convert the frame from BGR to RGB before processing with MediaPipe?
   A1. MediaPipe expects input images in RGB format, while OpenCV captures images in BGR format.
Q2. What is the purpose of setting static_image_mode to False in FaceMesh?
   A2. It allows the module to treat the input as a continuous video stream, optimizing tracking performance.
Q3. How can you modify the code to track multiple faces?
   A3. By setting max_num_faces to a value greater than 1, you enable detection and tracking of multiple faces.
"""

#%% Import required libraries
import cv2                      # OpenCV for video capture and image processing
import mediapipe as mp          # MediaPipe for face mesh detection and drawing utilities

#%% Initialize MediaPipe Face Mesh

# Load the FaceMesh module from MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Create a FaceMesh object with specified settings:
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,           # False for continuous video stream (optimized for video)
    max_num_faces=1,                   # Detect only one face (set >1 to track multiple faces)
    min_detection_confidence=0.5,      # Minimum confidence threshold for initial face detection
    min_tracking_confidence=0.5        # Minimum confidence threshold for tracking facial landmarks
)

# Initialize MediaPipe drawing utilities for rendering the face mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#%% Initialize the webcam

cap = cv2.VideoCapture(0)  # Open the default webcam (device index 0)

# Check if the webcam is accessible; if not, exit the script
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

#%% Main processing loop

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the image from BGR (OpenCV default) to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame using MediaPipe FaceMesh to detect facial landmarks
    results = face_mesh.process(rgb_frame)

    # If facial landmarks were detected, iterate through each detected face
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh tessellation (triangular mesh over the face)
            mp_drawing.draw_landmarks(
                image=frame,  # Draw on the original BGR frame
                landmark_list=face_landmarks,  # The detected facial landmarks
                connections=mp_face_mesh.FACEMESH_TESSELATION,  # Use tessellation connections for mesh
                landmark_drawing_spec=None,  # Do not draw individual landmark dots
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Draw additional contours (e.g., jawline, lips, eyebrows) over the face
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,  # Use predefined contour connections
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

    # Display the processed frame with the drawn face mesh in a window
    cv2.imshow('Mediapipe Face Mesh', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

#%% Clean up

cap.release()              # Release the webcam resource
cv2.destroyAllWindows()    # Close all OpenCV windows
