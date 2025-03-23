#!/usr/bin/env python3
"""
Top-Level Explanation:
This script implements real-time hand landmark detection using MediaPipe’s Hand Landmarker.
It captures video from a webcam, processes each frame to detect hand landmarks, and then visualizes the detection by drawing annotations on the frame.
In this example, it specifically highlights the thumb tip and index finger tip, and checks if the thumb is raised (thumb up gesture).
This demo is useful for understanding how to integrate MediaPipe models into real-time applications and can help answer lab questions about model initialization, image processing, and gesture detection.
Reference: https://github.com/googlesamples/mediapipe/tree/main/examples/hand_landmarker/raspberry_pi
"""

import cv2  # OpenCV for video capture and image processing.
import mediapipe as mp  # MediaPipe framework for hand landmark detection.
from mediapipe.tasks import python  # Import the Python wrapper for MediaPipe tasks.
from mediapipe.tasks.python import vision  # Import the Vision API for the Hand Landmarker.
 
#%% Parameters
numHands = 2  # Number of hands to be detected.
model = 'hand_landmarker.task'  # Path to the hand landmark detection model.
# Download using: wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

minHandDetectionConfidence = 0.5  # Minimum confidence threshold for hand detection.
minHandPresenceConfidence = 0.5     # Minimum confidence threshold for hand presence.
minTrackingConfidence = 0.5         # Minimum confidence threshold for hand tracking.
frameWidth = 640   # Desired width of the video frame.
frameHeight = 480  # Desired height of the video frame.

# Visualization parameters for text display.
MARGIN = 10          # Margin for drawing text.
FONT_SIZE = 1        # Font size for text.
FONT_THICKNESS = 1   # Font thickness for text.
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # Color for text (vibrant green).

#%% Create a HandLandmarker object.
# Set up the base options with the model path.
base_options = python.BaseOptions(model_asset_path=model)
# Configure the Hand Landmarker options.
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=numHands,
    min_hand_detection_confidence=minHandDetectionConfidence,
    min_hand_presence_confidence=minHandPresenceConfidence,
    min_tracking_confidence=minTrackingConfidence
)
# Create the hand landmarker from the options.
detector = vision.HandLandmarker.create_from_options(options)
# Q: Why do we need to set multiple confidence thresholds?
# A: These thresholds help ensure that the model only processes hands with sufficient detection, presence, and tracking quality.

#%% OpenCV Video Capture and Frame Analysis
cap = cv2.VideoCapture(0)  # Open the default webcam.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)   # Set the frame width.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight) # Set the frame height.
# Q: What does setting frame width and height achieve?
# A: It ensures that the captured frames match the expected input size for the model.

# Check if the webcam is opened correctly.
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Process frames until the 'q' key is pressed.
while True:
    try:
        # Capture one frame from the webcam.
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip the frame horizontally to mimic a mirror view.
        frame = cv2.flip(frame, 1)
        
        # Convert the image from BGR (OpenCV default) to RGB.
        # The model requires images in RGB format.
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Wrap the RGB image in a MediaPipe Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Run the hand landmarker on the image.
        detection_result = detector.detect(mp_image)
        # Q: What does detector.detect() return?
        # A: It returns a result object containing hand landmarks and other detection details.
        
        # Extract the list of hand landmarks from the detection result.
        hand_landmarks_list = detection_result.hand_landmarks
        # (Optional) handedness_list could be used to determine which hand is detected.
        # handedness_list = detection_result.handedness
        
        # Loop through each detected hand to visualize landmarks and gestures.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Detect the thumb tip by using index 4 (as per MediaPipe's hand landmark model).
            x_thumb = int(hand_landmarks[4].x * frame.shape[1])
            y_thumb = int(hand_landmarks[4].y * frame.shape[0])
            cv2.circle(frame, (x_thumb, y_thumb), 5, (0, 255, 0), -1)  # Draw a green circle.
            
            # Detect the index finger tip using index 8.
            x_index = int(hand_landmarks[8].x * frame.shape[1])
            y_index = int(hand_landmarks[8].y * frame.shape[0])
            cv2.circle(frame, (x_index, y_index), 5, (0, 255, 0), -1)  # Draw a green circle.
            
            # Define a threshold for determining if the thumb is up.
            threshold = 0.1
            thumb_tip_y = hand_landmarks[4].y
            thumb_base_y = hand_landmarks[1].y  # Index 1 corresponds to the thumb base.
            thums_up = thumb_tip_y < thumb_base_y - threshold  # Thumb up condition.
            
            # If the thumb is up, overlay text "Thumb Up" on the frame.
            if thums_up:
                cv2.putText(frame, 'Thumb Up', (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE,
                            HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                # Q: How is the thumb up gesture detected?
                # A: By comparing the y-coordinates of the thumb tip and thumb base, ensuring the tip is significantly higher.

        # Display the annotated frame.
        cv2.imshow('Annotated Image', frame)
        
        # Exit the loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
    except KeyboardInterrupt:
        # Gracefully exit on keyboard interrupt.
        break

# Release the video capture object and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
# Q: Why is cleanup necessary?
# A: To free system resources and ensure the program terminates without leaving open windows or active hardware connections.
