# === Top-Level Explanation ===
# This script processes an input video to create a summarized output by filtering frames that contain a target object.
# It uses MediaPipe's EfficientDet TFLite model for object detection to identify the specified target (e.g., "cell phone").
# If a frame contains the target object (case-insensitive match), that frame is saved as an image and added to the summary video.
#
# Potential Lab Test Q&A:
# Q: What is the main purpose of this code?
# A: To process an input video, detect a specified object in each frame, and create a summary video containing only those frames.
#
# Q: How does the script decide whether a frame should be included in the summary?
# A: The script runs object detection on each frame and checks if any detected object's category matches the target object.
#
# Q: What are the benefits of saving matching frames as both video and individual images?
# A: Saving as video provides a continuous summary, while images allow for detailed inspection of specific frames.

import cv2                 # OpenCV for video reading, writing, and image processing
import mediapipe as mp     # MediaPipe for real-time machine learning, here used for object detection
import time                # Time module (not directly used here but useful for potential timestamping)
import os                  # OS module for file system operations (e.g., creating directories)

# Import MediaPipe's Python task wrappers for vision tasks (object detection)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === CONFIGURATION ===
model_path = 'efficientdet.tflite'           # Path to the downloaded TFLite object detection model
target_object = 'cell phone'                 # Target object name to filter for (case-insensitive)
input_video_path = 'input_video.mp4'         # Path to the input video to summarize
output_video_path = 'summary_output.mp4'     # Path to the summarized output video
save_frames_dir = 'filtered_frames'          # Optional folder to store individual matching frames as images

# === OBJECT DETECTION PARAMETERS ===
score_threshold = 0.3     # Minimum confidence score required for a detection to be considered valid
max_results = 5           # Maximum number of objects to detect per frame

# === PREPARE OUTPUT FOLDER ===
os.makedirs(save_frames_dir, exist_ok=True)  # Create folder to save matched frames if it doesn't already exist

# === SET UP DETECTOR ===
# Define model and detection options for MediaPipe's object detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,       # Set to VIDEO mode for processing frames from a video file
    max_results=max_results,                       # Limit the number of detected objects per frame
    score_threshold=score_threshold                # Set the confidence threshold for valid detections
)

# Create the object detector using the specified options
detector = vision.ObjectDetector.create_from_options(options)
# Q: Why use the VIDEO running mode instead of LIVE_STREAM?
# A: VIDEO mode is used for processing pre-recorded video frames sequentially, as opposed to real-time streaming.

# === OPEN INPUT VIDEO ===
cap = cv2.VideoCapture(input_video_path)  # Open the input video file for reading
if not cap.isOpened():
    raise IOError(f"Cannot open video: {input_video_path}")

# Retrieve video properties for use in output video writer
fps = int(cap.get(cv2.CAP_PROP_FPS))             # Frames per second of the input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # Width of each frame
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # Height of each frame

# === OUTPUT VIDEO WRITER ===
# Initialize a video writer to save the summarized video (only frames that match the target object)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for MP4 video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# === FRAME LOOP ===
frame_idx = 0       # Frame counter to track current frame index
matched_count = 0   # Counter for the number of frames that matched the target object

while cap.isOpened():
    success, frame = cap.read()  # Read a frame from the input video
    if not success:
        break  # Exit the loop if there are no more frames or if there's an error

    # Convert the frame from BGR (default in OpenCV) to RGB (expected by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap the RGB frame in MediaPipe's Image class (with SRGB color format)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run the object detection model on the current frame
    detection_result = detector.detect(mp_image)

    # Check if any detection in the frame matches the target object
    match_found = False
    for detection in detection_result.detections:
        category = detection.categories[0]  # Use the top category prediction for each detection
        if category.category_name.lower() == target_object.lower():
            match_found = True
            break  # Exit loop once a match is found

    if match_found:
        matched_count += 1

        # Write the matching frame to the summary output video
        out.write(frame)

        # Optionally, save the matching frame as a JPEG image for further review
        filename = os.path.join(save_frames_dir, f'frame_{frame_idx:04d}.jpg')
        cv2.imwrite(filename, frame)

    frame_idx += 1  # Increment frame counter

# === CLEANUP ===
cap.release()   # Release the input video stream
out.release()   # Finalize and save the output video file

# Final report
print(f"✅ Summarization complete. {matched_count} matching frames saved.")
print(f"🎞️ Output video saved to: {output_video_path}")
