#!/usr/bin/env python3
"""
Top-Level Explanation:
This script performs real-time object detection using MediaPipe's lightweight EfficientDet model.
It captures video frames from a webcam, processes each frame with the MediaPipe object detector running
in live stream mode, and overlays bounding boxes, labels, and confidence scores on detected objects.
This demo is useful for learning about model initialization, asynchronous detection, and real-time
visualization of object detection results in computer vision applications.
Reference: https://github.com/googlesamples/mediapipe/blob/main/examples/object_detection/raspberry_pi
"""

import cv2  # OpenCV for video capture and image processing.
import mediapipe as mp  # MediaPipe framework for ML-based vision tasks.
import time  # Time module for timestamps and measuring time.

from mediapipe.tasks import python  # Import the Python wrapper for MediaPipe tasks.
from mediapipe.tasks.python import vision  # Import the Vision API for object detection tasks.

#%% Parameters
maxResults = 5  # Maximum number of detected objects per frame.
scoreThreshold = 0.25  # Minimum confidence score required to consider a detection valid.
frameWidth = 640  # Width of the video capture frame.
frameHeight = 480  # Height of the video capture frame.
model = 'efficientdet.tflite'  # Path to the EfficientDet TFLite model.
# Q: Why choose EfficientDet model?
# A: EfficientDet is optimized for speed and accuracy in resource-constrained environments.

# Visualization parameters for drawing labels.
MARGIN = 10       # Margin in pixels for label placement.
ROW_SIZE = 30     # Vertical space in pixels between rows of text.
FONT_SIZE = 1     # Font size for the text.
FONT_THICKNESS = 1  # Font thickness for the text.
TEXT_COLOR = (0, 0, 0)  # Text color: black.

#%% Initialize results and define a callback function to save detection results.
detection_frame = None  # Placeholder for the frame with detection annotations.
detection_result_list = []  # List to store detection results asynchronously.

def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
    """
    Callback function to save detection results.
    
    Parameters:
      result: The object detector result containing detection information.
      unused_output_image: The processed image (unused in this case).
      timestamp_ms: Timestamp of the frame in milliseconds.
      
    The result is appended to a global list for further processing.
    """
    detection_result_list.append(result)

#%% Create an object detection model object.
# Set up base options with the model asset path.
base_options = python.BaseOptions(model_asset_path=model)
# Configure the object detector options.
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,  # Use live stream mode for real-time processing.
    max_results=maxResults,
    score_threshold=scoreThreshold,
    result_callback=save_result  # Callback to handle results asynchronously.
)
# Create the object detector from the specified options.
detector = vision.ObjectDetector.create_from_options(options)
# Q: What does asynchronous detection provide?
# A: It allows non-blocking processing, improving real-time performance.

#%% OpenCV Video Capture and Frame Analysis
cap = cv2.VideoCapture(0)  # Open the default webcam.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)    # Set the desired frame width.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)  # Set the desired frame height.
# Q: Why configure frame dimensions?
# A: The model might require specific dimensions for optimal performance.

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

        # Convert the frame from BGR (OpenCV format) to RGB (model requirement).
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Wrap the RGB image in a MediaPipe Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Keep a copy of the current frame for drawing detection results.
        current_frame = frame.copy()

        # Run object detection asynchronously using the model.
        # time.time_ns() // 1_000_000 converts the current time to milliseconds.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # If detection results have been received, process them.
        if detection_result_list:
            # Loop through each detection in the first result.
            for detection in detection_result_list[0].detections:
                # Retrieve the bounding box from the detection.
                bbox = detection.bounding_box
                start_point = (bbox.origin_x, bbox.origin_y)
                end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                # Draw a rectangle around the detected object using an orange color.
                cv2.rectangle(current_frame, start_point, end_point, (0, 165, 255), 3)

                # Extract the category information from the detection.
                category = detection.categories[0]
                category_name = category.category_name  # Detected object's label.
                probability = round(category.score, 2)    # Confidence score.
                result_text = f'{category_name} ({probability})'
                # Define the text location near the top-left of the bounding box.
                text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
                # Draw the label and score on the frame.
                cv2.putText(current_frame, result_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

            # Update the detection frame to be displayed.
            detection_frame = current_frame
            # Clear the detection results for the next frame.
            detection_result_list.clear()

        # Display the frame with detection overlays if available.
        if detection_frame is not None:
            cv2.imshow('object_detection', detection_frame)

        # Break the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Gracefully exit if a keyboard interrupt is detected.
        break

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
# Q: Why is cleanup important in video capture applications?
# A: To free up system resources and ensure that hardware and windows are properly released upon exit.
