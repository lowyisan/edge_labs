#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script demonstrates real-time object detection using MediaPipe's lightweight EfficientDet model.
It initializes the model using a TFLite file (efficientdet_lite0.tflite), captures video from the webcam,
and processes each frame to detect objects. Detected objects are visualized by drawing bounding boxes and labels
on the frame. This example is useful for lab tests covering real-time computer vision, TFLite model integration,
and live-stream processing.

# Potential Lab Test Q&A:
# Q: What is the purpose of using MediaPipe's EfficientDet model in this code?
# A: The model is used for real-time object detection. It identifies objects in video frames, providing
#    bounding boxes and confidence scores for each detected object.
#
# Q: Why is the detection run asynchronously?
# A: Running the detection asynchronously helps avoid blocking the main video capture loop,
#    ensuring smoother real-time performance.
#
# Q: How are the bounding boxes and labels drawn on the image?
# A: The code extracts the bounding box coordinates and category information from the detection result,
#    then uses OpenCV's drawing functions to overlay rectangles and text on the image.

Potential Lab Test Questions:
Q1. Why do we convert the frame from BGR to RGB before passing it to the object detector?
   A1. The TFLite model and MediaPipe expect the input image in RGB format, while OpenCV captures images in BGR.
Q2. What is the purpose of the result_callback parameter in the ObjectDetectorOptions?
   A2. It specifies a function (save_result) that is called asynchronously to store or process detection results.
Q3. How is the real-time detection result visualized on the frame?
   A3. The script draws bounding boxes and labels (category name and score) on the frame based on the detection results.
"""

#%% Import required libraries
import cv2                        # OpenCV for video capture and image processing
import mediapipe as mp            # MediaPipe for image processing and model inference
import time                       # For timing and timestamping

# Import the MediaPipe Tasks Python wrapper and vision module
from mediapipe.tasks import python  # Python wrapper for MediaPipe Tasks
from mediapipe.tasks.python import vision  # Vision API for object detection

#%% Parameters
maxResults = 5                   # Maximum number of detected objects per frame
scoreThreshold = 0.25            # Minimum confidence score to consider a detection valid
frameWidth = 640                 # Desired frame width for capture
frameHeight = 480                # Desired frame height for capture
model = 'efficientdet.tflite'    # Path to the TFLite EfficientDet model file

# Visualization parameters
MARGIN = 10           # Margin in pixels for text placement
ROW_SIZE = 30         # Height of the text row in pixels
FONT_SIZE = 1         # Font scale for text
FONT_THICKNESS = 1    # Thickness of the text
TEXT_COLOR = (0, 0, 0)  # Text color (black)

#%% Initialize results and callback for detection results
detection_frame = None         # Global variable to store the frame with detection results
detection_result_list = []     # List to store detection results asynchronously

def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
    """
    Callback function to save detection results.
    This function is called asynchronously when a detection result is ready.
    
    Parameters:
      result: The detection result containing detected objects.
      unused_output_image: The image output from the detection (unused here).
      timestamp_ms: Timestamp in milliseconds of the processed frame.
    """
    detection_result_list.append(result)

#%% Create an object detection model object.
# Initialize the object detection model using the TFLite model and set parameters
base_options = python.BaseOptions(model_asset_path=model)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,  # Set mode to live-stream for video processing
    max_results=maxResults,
    score_threshold=scoreThreshold,
    result_callback=save_result                # Set callback to store results asynchronously
)
detector = vision.ObjectDetector.create_from_options(options)

#%% OpenCV Video Capture and frame analysis
# Open the default webcam and set capture resolution as per model requirements
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

# Validate webcam connection
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#%% Main Processing Loop
# The loop will break on pressing the 'q' key
while True:
    try:
        # Capture one frame from the webcam
        ret, frame = cap.read() 
        # Flip the frame horizontally for a mirror-like view (optional)
        frame = cv2.flip(frame, 1)
        
        # Convert the frame from BGR (OpenCV) to RGB (MediaPipe requirement)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create a MediaPipe Image object from the RGB frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        current_frame = frame.copy()  # Copy the current frame for drawing detections
        
        # Run object detection asynchronously on the current frame.
        # The timestamp is provided in milliseconds.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)
        
        # If there are detection results available, process them
        if detection_result_list:
            # Process the first result in the list
            for detection in detection_result_list[0].detections:
                # Get the bounding box for the detected object
                bbox = detection.bounding_box
                start_point = (bbox.origin_x, bbox.origin_y)
                end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                # Draw the bounding box using an orange rectangle for visibility
                cv2.rectangle(current_frame, start_point, end_point, (0, 165, 255), 3)
            
                # Extract the top category and score for the detected object
                category = detection.categories[0]
                category_name = category.category_name
                probability = round(category.score, 2)
                result_text = f"{category_name} ({probability})"
                # Calculate text location with margins
                text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
                # Draw the label text on the frame
                cv2.putText(current_frame, result_text, text_location,
                            cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            
            # Set the detection_frame to the current frame with drawn detections
            detection_frame = current_frame
            # Clear the detection result list for next frame
            detection_result_list.clear()
    
        # If a detection frame exists, show it in a window
        if detection_frame is not None:
            cv2.imshow('object_detection', detection_frame)
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
    except KeyboardInterrupt:
        break

#%% Cleanup
cap.release()            # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
