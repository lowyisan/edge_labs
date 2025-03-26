#!/usr/bin/env python3
"""
Top-Level Explanation:
This script implements real-time hand gesture recognition using MediaPipe’s gesture recognizer model.
It captures video from a webcam, processes each frame to detect hand gestures, draws hand landmarks,
and overlays gesture classification results on the video feed.
The code uses MediaPipe Tasks’ Python API to run a pre-trained gesture recognizer model in live stream mode.
This demo is useful for understanding real-time gesture detection and can help answer questions related
to model initialization, asynchronous processing, landmark drawing, and result visualization in computer vision.
Reference: https://github.com/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/raspberry_pi/
"""

import cv2                     # OpenCV for video capture and image processing.
import mediapipe as mp         # MediaPipe framework for hand gesture recognition.
import time                    # Time module for time stamps and timing operations.

from mediapipe.tasks import python            # Python wrapper for MediaPipe tasks.
from mediapipe.tasks.python import vision       # Vision API for gesture recognition and configuration.
from mediapipe.framework.formats import landmark_pb2  # For handling landmark data in protobuf format.

# Initialize MediaPipe hands drawing utilities for visualization.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#%% Parameters
numHands = 2  # Number of hands to be detected.
model = 'gesture_recognizer.task'  
# Model for hand gesture detection.
# Download using: wget -O gesture_recognizer.task -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

# Set thresholds for hand detection and tracking.
minHandDetectionConfidence = 0.5  
minHandPresenceConfidence = 0.5
minTrackingConfidence = 0.5

# Define the capture resolution as required by the model.
frameWidth = 640
frameHeight = 480

# Visualization parameters for drawing text on the frame.
row_size = 50        # Pixel height for row spacing.
left_margin = 24     # Pixel left margin.
text_color = (0, 0, 0) # Black color for text.
font_size = 1        # Font size.
font_thickness = 1   # Font thickness.

# Label box parameters (for gesture text).
label_text_color = (255, 255, 255)  # White text.
label_font_size = 1
label_thickness = 2

#%% Initialize results and a callback for appending recognition results.
recognition_frame = None         # Placeholder for the frame with drawn results.
recognition_result_list = []     # List to store recognition results from the gesture recognizer.

def save_result(result: vision.GestureRecognizerResult,
                unused_output_image: mp.Image, timestamp_ms: int):
    """
    Callback function to save gesture recognition results.
    
    Parameters:
      result: The gesture recognition result from the model.
      unused_output_image: The processed image (not used here).
      timestamp_ms: Timestamp of the frame in milliseconds.
      
    Appends the result to the global recognition_result_list.
    """
    recognition_result_list.append(result)

#%% Create a Hand Gesture Control object.
# Initialize the gesture recognizer model with specified options.
base_options = python.BaseOptions(model_asset_path=model)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,  # Live stream mode for real-time processing.
    num_hands=numHands,
    min_hand_detection_confidence=minHandDetectionConfidence,
    min_hand_presence_confidence=minHandPresenceConfidence,
    min_tracking_confidence=minTrackingConfidence,
    result_callback=save_result  # Callback function to handle asynchronous results.
)
recognizer = vision.GestureRecognizer.create_from_options(options)
# Q: Why use a callback for results?
# A: The asynchronous processing allows non-blocking gesture recognition, improving real-time performance.

#%% OpenCV Video Capture and Frame Analysis
cap = cv2.VideoCapture(0)  # Open the default webcam.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)   # Set frame width.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight) # Set frame height.

# Check if the webcam is opened correctly.
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Process frames continuously until 'q' is pressed.
while True:
    try:
        # Capture one frame from the webcam.
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip the frame horizontally to match the typical mirror view.
        frame = cv2.flip(frame, 1)
        
        # Convert the image from BGR (OpenCV default) to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Wrap the image in a MediaPipe Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Keep a copy of the current frame for drawing results.
        current_frame = frame.copy()
        
        # Run hand gesture recognition asynchronously.
        # time.time_ns() // 1_000_000 converts current time to milliseconds.
        recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)
        
        # If any recognition result is available, process and draw them.
        if recognition_result_list:
            # Process results for each detected hand.
            for hand_index, hand_landmarks in enumerate(recognition_result_list[0].hand_landmarks):
                # Calculate the bounding box of the hand based on landmark coordinates.
                x_min = min([landmark.x for landmark in hand_landmarks])
                y_min = min([landmark.y for landmark in hand_landmarks])
                y_max = max([landmark.y for landmark in hand_landmarks])
    
                # Convert normalized coordinates (0-1) to pixel values.
                frame_height, frame_width = current_frame.shape[:2]
                x_min_px = int(x_min * frame_width)
                y_min_px = int(y_min * frame_height)
                y_max_px = int(y_max * frame_height)
    
                # Get gesture classification results.
                if recognition_result_list[0].gestures:
                    gesture = recognition_result_list[0].gestures[hand_index]
                    category_name = gesture[0].category_name  # Name of the recognized gesture.
                    score = round(gesture[0].score, 2)         # Confidence score.
                    result_text = f'{category_name} ({score})'
    
                    # Compute the size of the text for display.
                    text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX,
                                                label_font_size, label_thickness)[0]
                    text_width, text_height = text_size
    
                    # Calculate the text position (above the hand bounding box).
                    text_x = x_min_px
                    text_y = y_min_px - 10  # Adjust vertical position.
    
                    # Ensure the text is within frame boundaries.
                    if text_y < 0:
                        text_y = y_max_px + text_height
    
                    # Draw the text (gesture name and score) on the frame.
                    cv2.putText(current_frame, result_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                                label_text_color, label_thickness, cv2.LINE_AA)
    
                # Draw hand landmarks on the frame.
                # Convert hand landmarks to a protobuf format required by MediaPipe drawing utilities.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ])
                mp_drawing.draw_landmarks(
                    current_frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
    
            # Update the frame to be displayed with recognition results.
            recognition_frame = current_frame
            # Clear the result list for the next frame.
            recognition_result_list.clear()
    
        # Display the frame with gesture recognition overlays.
        if recognition_frame is not None:
            cv2.imshow('gesture_recognition', recognition_frame)
        
        # Exit loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
    except KeyboardInterrupt:
        # Graceful exit on keyboard interrupt (e.g., Ctrl+C).
        break

# Release the webcam and destroy all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
# Q: Why is cleanup important?
# A: It frees system resources and closes display windows, ensuring the application terminates properly.
