#%% Top-Level Explanation
# This code performs real-time hand gesture recognition using a webcam.
# It captures video frames via OpenCV, processes them with MediaPipe's gesture recognition model,
# and overlays the detected hand landmarks and gesture labels on the video feed.
# Key features include asynchronous result processing via a callback function,
# customizable detection/tracking parameters, and a real-time display loop.
#
# Potential Lab Test Q&A:
# Q: What is the purpose of using MediaPipe in this code?
# A: MediaPipe is used to provide a pre-trained gesture recognition model and utilities
#    to detect hand landmarks and gestures in real-time.
#
# Q: How does the asynchronous callback mechanism improve performance?
# A: It allows the recognition results to be processed separately from the main capture loop,
#    ensuring smooth real-time video processing.

#%% Reference: https://github.com/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/raspberry_pi/
# Download hand gesture detector model
# wget -O gesture_recognizer.task -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

import cv2                     # OpenCV for video capture and image processing
import mediapipe as mp         # MediaPipe for gesture recognition
import time                    # Time module for timestamping frames

# Import MediaPipe's Python wrapper and vision-specific APIs
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 
from mediapipe.framework.formats import landmark_pb2  # Used for converting landmark data to protobuf for drawing

# Predefined modules for drawing hand connections and landmarks
mp_hands = mp.solutions.hands                     # Provides hand detection and tracking functionalities
mp_drawing = mp.solutions.drawing_utils           # Utility functions for drawing landmarks and connections
mp_drawing_styles = mp.solutions.drawing_styles     # Predefined styles for drawing, improving visualization

#%% Parameters

numHands = 2  
# Maximum number of hands to detect simultaneously.
# Q: What happens if numHands is increased?
# A: More hands can be detected but at the cost of increased computational load.

model = 'gesture_recognizer.task'  
# Path to the TFLite gesture recognition model file.
# Q: Why is this model file important?
# A: It contains the pre-trained weights and parameters necessary for recognizing gestures.

minHandDetectionConfidence = 0.5  
# Minimum confidence level required for initial hand detection.
# Q: What does lowering this value imply?
# A: It could detect hands with lower confidence but might lead to more false positives.

minHandPresenceConfidence = 0.5  
# Ensures that a detected hand is still present in subsequent frames.
# Q: Why is this important?
# A: It helps reduce flickering in detection when a hand momentarily becomes less visible.

minTrackingConfidence = 0.5  
# Minimum confidence required to continue tracking a hand.
# Q: What effect does this have on tracking?
# A: It ensures the tracking remains stable; too high might drop hands quickly, too low may allow inaccuracies.

frameWidth = 640
frameHeight = 480
# Resolution settings for the webcam capture.
# Q: How do these settings affect the performance?
# A: Higher resolution gives more detail but requires more processing power.

# Visualization parameters
row_size = 50           # Vertical spacing between text lines on the display
left_margin = 24        # Left margin for text display
text_color = (0, 0, 0)  # Text color in BGR (black)
font_size = 1           # Font size for general text
font_thickness = 1      # Font thickness for general text

# Label box parameters for displaying gesture labels
label_text_color = (255, 255, 255)  # Label text color in BGR (white)
label_font_size = 1                 # Font size for labels
label_thickness = 2                 # Thickness for label text

#%% Initialize results and callback storage.
recognition_frame = None            # This will hold the annotated frame for display
recognition_result_list = []        # List to store recognition results returned asynchronously

# Callback function called when the model produces a result
def save_result(result: vision.GestureRecognizerResult, unused_output_image: mp.Image, timestamp_ms: int):
    # Append the gesture recognition result to the global result list
    recognition_result_list.append(result)
    # Q: Why use a callback function here?
    # A: It allows asynchronous processing of recognition results, crucial for live stream performance.

#%% Create a Hand Gesture Control object.

# Initialize base model options with the path to the gesture recognition model
base_options = python.BaseOptions(model_asset_path=model)

# Set up the gesture recognizer options with detection and tracking parameters
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,  # Enables real-time (live stream) inference
    num_hands=numHands,
    min_hand_detection_confidence=minHandDetectionConfidence,
    min_hand_presence_confidence=minHandPresenceConfidence,
    min_tracking_confidence=minTrackingConfidence,
    result_callback=save_result  # Use the callback to handle asynchronous results
)
# Q: What is the benefit of LIVE_STREAM mode?
# A: It processes frames in real time, which is essential for interactive applications.

# Create the gesture recognizer instance using the options defined above
recognizer = vision.GestureRecognizer.create_from_options(options)

#%% OpenCV Video Capture and Frame Analysis
cap = cv2.VideoCapture(0)  # Open the default webcam

# Set the capture resolution as specified by frameWidth and frameHeight
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

# Verify that the webcam is successfully opened
if not cap.isOpened():
    raise IOError("Cannot open webcam")
    # Q: What does this check do?
    # A: It ensures that the webcam is accessible before proceeding.

# Main loop: Process video frames until 'q' is pressed
while True:
    try:
        # Capture a frame from the webcam
        ret, frame = cap.read() 
        # Q: What does cap.read() return?
        # A: A boolean (ret) indicating success and the captured frame (frame).

        # Flip the frame horizontally for a natural mirror view
        frame = cv2.flip(frame, 1)

        # Convert the image from BGR (OpenCV default) to RGB (required by the model)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Wrap the RGB frame into MediaPipe's image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Store the current frame to annotate later with landmarks and labels
        current_frame = frame

        # Run gesture recognition asynchronously using the current timestamp in milliseconds
        recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

        # If results are available from the callback, process them
        if recognition_result_list:
            # Loop over each detected hand and its associated landmarks
            for hand_index, hand_landmarks in enumerate(recognition_result_list[0].hand_landmarks):

                # --- Bounding Box Computation ---
                # Calculate minimum and maximum landmark positions to draw a bounding box near the hand
                x_min = min([landmark.x for landmark in hand_landmarks])
                y_min = min([landmark.y for landmark in hand_landmarks])
                y_max = max([landmark.y for landmark in hand_landmarks])

                # Convert normalized coordinates (0 to 1) to actual pixel values
                frame_height, frame_width = current_frame.shape[:2]
                x_min_px = int(x_min * frame_width)
                y_min_px = int(y_min * frame_height)
                y_max_px = int(y_max * frame_height)

                # --- Display Gesture Label ---
                # Check if gesture results exist and retrieve the top result for the current hand
                if recognition_result_list[0].gestures:
                    gesture = recognition_result_list[0].gestures[hand_index]
                    category_name = gesture[0].category_name  # Name of the detected gesture
                    score = round(gesture[0].score, 2)         # Confidence score (rounded)
                    result_text = f'{category_name} ({score})'

                    # Get text dimensions for proper label positioning
                    text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, label_font_size, label_thickness)[0]
                    text_width, text_height = text_size

                    # Position the label slightly above the detected hand's bounding box
                    text_x = x_min_px
                    text_y = y_min_px - 10  # Offset upward

                    # Ensure that the text does not get drawn off-screen
                    if text_y < 0:
                        text_y = y_max_px + text_height

                    # Draw the gesture label on the frame
                    cv2.putText(current_frame, result_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_DUPLEX,
                                label_font_size,
                                label_text_color,
                                label_thickness,
                                cv2.LINE_AA)

                # --- Draw Landmarks & Connections on the Frame ---
                # Convert detected hand landmarks into a protobuf format suitable for drawing
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ])

                # Use MediaPipe's drawing utility to overlay landmarks and connections on the frame
                mp_drawing.draw_landmarks(
                    current_frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Update the frame with the recognition annotations for display
            recognition_frame = current_frame

            # Clear the recognition results list to prepare for the next frame
            recognition_result_list.clear()

        # Display the annotated frame if available
        if recognition_frame is not None:
            cv2.imshow('gesture_recognition', recognition_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Graceful exit if interrupted via Ctrl+C
        break

# Release the webcam and close the OpenCV window when done
cap.release()
cv2.destroyAllWindows()
