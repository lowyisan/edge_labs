#%% Top-Level Explanation
# This code performs real-time hand landmark detection using a webcam. It utilizes MediaPipe's
# HandLandmarker model to identify hand landmarks and then processes those landmarks to:
#   - Visualize all 21 hand landmarks with blue dots.
#   - Detect specific gestures (e.g., "Thumb Up").
#   - Count the number of raised fingers.
# The annotated results (landmarks, gesture labels, and finger count) are overlaid on the video stream.
#
# Potential Lab Test Q&A:
# Q: What is the purpose of the HandLandmarker model in this code?
# A: The model detects and provides precise hand landmark locations, which are used to identify gestures and count raised fingers.
#
# Q: How does the code determine if a "Thumb Up" gesture is present?
# A: It compares the y-coordinate of the thumb tip (landmark 4) with that of the thumb base (landmark 1) and
#    checks if the thumb tip is significantly higher than the base.
#
# Q: What is the significance of counting raised fingers in the code?
# A: Counting raised fingers can be used as a simple gesture-based control or feedback mechanism,
#    demonstrating practical applications of hand landmark detection.

#%% Reference:
# https://github.com/googlesamples/mediapipe/tree/main/examples/hand_landmarker/raspberry_pi
# Model file can be downloaded using:
# wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

import cv2                    # OpenCV: for video capture, image processing, and drawing annotations
import mediapipe as mp        # MediaPipe: for hand landmark detection tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision  # Vision-specific APIs for hand landmark detection

#%% Parameters

numHands = 2  
# Maximum number of hands to detect in each frame.
# ↑: Detecting more hands increases processing time.
# ↓: Fewer hands detected result in faster processing.

model = 'hand_landmarker.task'  
# Path to the downloaded TFLite model for hand landmark detection.
# Ensure the model file exists or adjust the path as needed.

minHandDetectionConfidence = 0.5  
# Minimum confidence threshold for hand detection.
# Range: 0.0 to 1.0
# ↑: More strict detection, but may miss some hands.
# ↓: Looser detection, possibly increasing false positives.

minHandPresenceConfidence = 0.5  
# Threshold ensuring the hand remains visible after initial detection.
# ↑: Results are more stable.
# ↓: May cause flickering when a hand is partially occluded.

minTrackingConfidence = 0.5  
# Confidence threshold for tracking the hand across successive frames.
# ↑: Smoother and more stable tracking (but might be slower).
# ↓: May reduce accuracy during motion or occlusion.

frameWidth = 640   # Width of the webcam frame in pixels.
frameHeight = 480  # Height of the webcam frame in pixels.

# Visualization parameters for drawing overlays on the image.
MARGIN = 10                   # Margin (in pixels) for positioning text/labels.
FONT_SIZE = 1                 # Font size for overlay text.
FONT_THICKNESS = 1            # Thickness of the overlay text.
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # Text color (BGR): vibrant green for labels.

#%% Create a HandLandmarker object

# Load the model using MediaPipe's BaseOptions.
base_options = python.BaseOptions(model_asset_path=model)

# Define options for the hand landmarker, including detection and tracking parameters.
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=numHands,  # Number of hands to track.
    min_hand_detection_confidence=minHandDetectionConfidence,
    min_hand_presence_confidence=minHandPresenceConfidence,
    min_tracking_confidence=minTrackingConfidence
)

# Instantiate the hand detector from the defined options.
detector = vision.HandLandmarker.create_from_options(options)
# Q: Why is it important to set thresholds like detection and tracking confidence?
# A: They balance the accuracy and responsiveness of detection; stricter thresholds reduce false positives,
#    while looser thresholds ensure hands are not missed under varying conditions.

#%% OpenCV Webcam Capture Setup

# Open the default webcam using device index 0.
cap = cv2.VideoCapture(0)

# Set the webcam resolution to the specified width and height.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

# Validate the webcam connection.
if not cap.isOpened():
    raise IOError("Cannot open webcam")
    # Q: What happens if the webcam cannot be opened?
    # A: The program raises an IOError to alert the user and halt execution.

#%% Main Loop: Process Video Frames
while True:
    try:
        # Capture a frame from the webcam.
        ret, frame = cap.read()
        # Q: What do the variables 'ret' and 'frame' represent?
        # A: 'ret' is a boolean indicating if the frame was captured successfully;
        #    'frame' contains the image data.

        # Flip the frame horizontally to provide a mirror effect, which is more intuitive.
        frame = cv2.flip(frame, 1)

        # Convert the frame from BGR (OpenCV default) to RGB (expected by TFLite model).
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the RGB image in MediaPipe's Image format.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run the detector to identify hands and their landmarks.
        detection_result = detector.detect(mp_image)

        # Extract the list of hand landmarks from the detection result.
        hand_landmarks_list = detection_result.hand_landmarks
        # Optionally, handedness_list could be used to differentiate left/right hands.
        # handedness_list = detection_result.handedness

        total_fingers_up = 0  # Initialize a counter for the total number of raised fingers.

        # Process each detected hand.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # ---------------- [MODIFICATION] Show all 21 landmarks ----------------
            # Loop through each of the 21 landmarks and draw a small blue dot.
            for i, landmark in enumerate(hand_landmarks):
                # Convert normalized landmark coordinates to pixel coordinates.
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Blue dot

            # ---------------- Detect and Draw Thumb Tip ----------------
            # Landmark index 4 corresponds to the thumb tip.
            x = int(hand_landmarks[4].x * frame.shape[1])
            y = int(hand_landmarks[4].y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green dot for thumb tip

            # ---------------- Detect and Draw Index Finger Tip ----------------
            # Landmark index 8 corresponds to the index finger tip.
            x = int(hand_landmarks[8].x * frame.shape[1])
            y = int(hand_landmarks[8].y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green dot for index finger tip

            # ---------------- Thumb Up Gesture Detection ----------------
            # Compare the y-coordinates of the thumb tip (index 4) and thumb base (index 1).
            # If the thumb tip is significantly above the base, consider it a "Thumb Up".
            threshold = 0.1  # Minimum vertical distance required (in normalized units).
            thumb_tip_y = hand_landmarks[4].y
            thumb_base_y = hand_landmarks[1].y

            # In image coordinates, a lower y-value is higher up in the image.
            thums_up = thumb_tip_y < (thumb_base_y - threshold)

            # Display the label "Thumb Up" if the gesture is detected.
            if thums_up:
                cv2.putText(frame, 'Thumb Up', (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE,
                            HANDEDNESS_TEXT_COLOR,
                            FONT_THICKNESS,
                            cv2.LINE_AA)

            # ---------------- [MODIFICATION] Count Raised Fingers ----------------
            # Compare each fingertip with its corresponding PIP joint to determine if a finger is raised.
            # Landmark pairs: (thumb: 4-2), (index: 8-6), (middle: 12-10), (ring: 16-14), (pinky: 20-18)
            tips_ids = [4, 8, 12, 16, 20]
            base_ids = [2, 6, 10, 14, 18]

            fingers_up = 0  # Count of raised fingers for the current hand.
            for tip_id, base_id in zip(tips_ids, base_ids):
                tip = hand_landmarks[tip_id]
                base = hand_landmarks[base_id]

                # For the thumb, compare x-coordinates because it points sideways.
                if tip_id == 4:
                    if tip.x > base.x:  # Adjust for right hand; may need reversal for left hand.
                        fingers_up += 1
                else:
                    if tip.y < base.y:  # If the fingertip is higher than the joint, finger is raised.
                        fingers_up += 1

            # Accumulate raised fingers count across all detected hands.
            total_fingers_up += fingers_up

        # ---------------- [MODIFICATION] Display Total Finger Count ----------------
        # Overlay the total number of raised fingers on the frame.
        cv2.putText(frame, f'Fingers: {total_fingers_up}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SIZE + 0.5,
                    (0, 255, 255), 2, cv2.LINE_AA)  # Yellow text for clear visibility

        # Show the annotated frame in a window.
        cv2.imshow('Annotated Image', frame)

        # Exit the loop when the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Handle a graceful exit when interrupted (e.g., Ctrl+C).
        break

# Release the webcam and close all OpenCV windows when done.
cap.release()
cv2.destroyAllWindows()
