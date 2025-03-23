#!/usr/bin/env python3
"""
Top-Level Explanation:
This script uses MediaPipe's Face Mesh solution to detect and draw facial landmarks in real-time from a webcam feed.
It captures video frames, processes them with MediaPipe to identify facial landmarks, and overlays the mesh and contours on the face.
The demo is useful for understanding real-time face detection, landmark extraction, and drawing utilities in computer vision applications.
"""

import cv2                     # OpenCV for accessing the webcam and image processing.
import mediapipe as mp         # MediaPipe for efficient real-time face landmark detection.

#%% Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh  # Access the Face Mesh module from MediaPipe.
# Create a FaceMesh object with specified parameters.
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,          # False for video stream mode (processes consecutive frames).
    max_num_faces=1,                  # Detect at most one face.
    min_detection_confidence=0.5,     # Minimum confidence value for face detection.
    min_tracking_confidence=0.5       # Minimum confidence value for tracking the face landmarks.
)
# Q: Why set static_image_mode to False?
# A: Because we're processing a video stream, so we want the model to track faces across frames rather than detecting each frame independently.

# Initialize MediaPipe drawing utilities for visualization.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#%% Open the camera feed using OpenCV
cap = cv2.VideoCapture(0)  # Open the default webcam (index 0).
# Q: What happens if the camera isn't available?
# A: The program will print an error message and exit.

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

#%% Process each frame from the webcam
while cap.isOpened():
    ret, frame = cap.read()  # Capture a frame from the webcam.
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the captured frame from BGR (OpenCV default) to RGB (MediaPipe requires RGB).
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Q: Why convert BGR to RGB?
    # A: MediaPipe models are trained on RGB images, so conversion is needed for proper processing.

    # Process the RGB frame to detect face landmarks.
    results = face_mesh.process(rgb_frame)

    # Draw facial landmarks on the original frame if any are detected.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the tesselation (mesh) connections.
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # Draw the contour connections (outline of the face).
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            # Q: What do FACEMESH_TESSELATION and FACEMESH_CONTOURS represent?
            # A: Tesselation shows the full mesh of facial landmarks, while contours represent the outer shape of the face.

    # Display the frame with the drawn facial landmarks.
    cv2.imshow('Mediapipe Face Mesh', frame)

    # Exit the loop when 'q' key is pressed.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows.
cap.release()
cv2.destroyAllWindows()
# Q: Why is it important to release the camera and destroy windows?
# A: To free system resources and ensure that the application closes gracefully.
