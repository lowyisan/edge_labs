#%% Top-Level Explanation
# This code performs real-time object detection using a webcam and MediaPipe's lightweight EfficientDet model.
# It captures video frames via OpenCV, processes them asynchronously with the MediaPipe object detector,
# and overlays bounding boxes, labels, and confidence scores on detected objects. The model detects up to a
# specified number of objects per frame, and the detection results are displayed in a live video window.
#
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

#%% Reference:
# https://github.com/googlesamples/mediapipe/blob/main/examples/object_detection/raspberry_pi
# Download lightweight TFLite EfficientDet model
# wget -q -O efficientdet.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite

import cv2                     # OpenCV: for video capture, image processing, and drawing annotations
import mediapipe as mp         # MediaPipe: for real-time machine learning solutions
import time                    # Time module: used for timestamping frames

# Import MediaPipe's Python task wrappers for vision tasks (object detection, etc.)
from mediapipe.tasks import python    # Base API for task configuration
from mediapipe.tasks.python import vision  # Vision-specific modules for object detection

#%% Parameters

maxResults = 5                # Maximum number of objects to detect in a frame
scoreThreshold = 0.25         # Minimum confidence score to consider a detection valid
frameWidth = 640              # Width of captured video frame
frameHeight = 480             # Height of captured video frame
model = 'efficientdet.tflite' # Path to the downloaded object detection model

# Visualization parameters for drawing bounding boxes and text labels
MARGIN = 10                 # Margin from bounding box for text display
ROW_SIZE = 30               # Vertical spacing of text lines
FONT_SIZE = 1               # Font size for drawing labels
FONT_THICKNESS = 1          # Thickness of label text
TEXT_COLOR = (0, 0, 0)      # Black color for label text (BGR format)

#%% Initializing results and result callback

detection_frame = None          # Will store the annotated output frame for display
detection_result_list = []      # List to collect detection results from the callback

# Callback function to save detection results into detection_result_list
def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
    # Append the result to the global list for processing in the main loop
    detection_result_list.append(result)
    # Q: Why is a callback function used here?
    # A: The callback allows asynchronous processing of detection results, improving real-time performance.

#%% Create an object detection model object

# Set the base model options using the path to the model file
base_options = python.BaseOptions(model_asset_path=model)

# Configure the object detection task with the desired parameters
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,  # Enable real-time streaming mode
    max_results=maxResults,                       # Limit the number of detected objects per frame
    score_threshold=scoreThreshold,               # Confidence threshold for valid detections
    result_callback=save_result                   # Set callback to handle asynchronous detection results
)

# Create an instance of the object detector using the defined options
detector = vision.ObjectDetector.create_from_options(options)
# Q: What does running_mode LIVE_STREAM mean?
# A: It allows the detector to process frames in real time as they are captured from the webcam.

#%% OpenCV Webcam Setup

# Open the default webcam (device index 0)
cap = cv2.VideoCapture(0)

# Set the video frame resolution as expected by the model
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

# Validate that the webcam is connected and accessible
if not cap.isOpened():
    raise IOError("Cannot open webcam")
    # Q: What happens if the webcam is not accessible?
    # A: The program raises an IOError and stops execution, alerting the user.

#%% Main Loop for Continuous Video Processing

while True:
    try:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        # Q: What do ret and frame represent?
        # A: 'ret' is a boolean indicating whether the frame was captured successfully, and 'frame' is the image data.

        # Flip the frame horizontally to create a mirror view for intuitive interaction
        frame = cv2.flip(frame, 1)

        # Convert the frame from BGR (OpenCV format) to RGB (expected by the TFLite model)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the RGB image in a MediaPipe Image object (with SRGB color format)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Keep a copy of the current frame to draw detection results on
        current_frame = frame

        # Run the object detector asynchronously; timestamp in milliseconds
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Process detection results if available
        if detection_result_list:
            # Iterate through each detected object in the first result (assuming one result per frame)
            for detection in detection_result_list[0].detections:
                # Extract the bounding box coordinates from the detection result
                bbox
