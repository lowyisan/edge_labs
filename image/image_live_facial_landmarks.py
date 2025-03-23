#!/usr/bin/env python3
"""
Top-Level Explanation:
This script uses the face_recognition library along with OpenCV to perform real-time face detection and facial landmark extraction.
It captures video frames from a webcam, detects faces in each frame, draws rectangles around them, and overlays facial landmarks (such as eyes, nose, mouth, etc.)
on the detected faces. This demonstration is useful for understanding real-time face recognition, detection, and landmark extraction, which are common
tasks in computer vision applications. Reference: https://github.com/ageitgey/face_recognition
"""

import cv2  # OpenCV library for video capture and image processing.
import face_recognition  # Library for face detection and recognition.
                         # Q: Why use face_recognition?
                         # A: It provides an easy-to-use interface for detecting faces and extracting facial features with high accuracy.

#%% OpenCV Video Capture and Frame Analysis
cap = cv2.VideoCapture(0)  # Open the default webcam (device index 0).
# Q: What does cv2.VideoCapture(0) do?
# A: It initializes video capture from the system's default camera.

# Check if the webcam is opened correctly.
if not cap.isOpened():
    raise IOError("Cannot open webcam")  # Raise an error if the webcam is not accessible.

# Process video frames continuously until 'q' is pressed.
while True:
    try:
        # Capture one frame from the webcam.
        ret, frame = cap.read()
        # Q: What do 'ret' and 'frame' represent?
        # A: 'ret' is a boolean that indicates whether the frame was captured successfully, and 'frame' is the captured image.

        # Resize the frame to 256x256 pixels for faster processing.
        frame = cv2.resize(frame, (256, 256))
        # Q: How does resizing help?
        # A: Smaller images require less processing time, speeding up face detection and landmark extraction.

        # Extract face locations in the frame using face_recognition.
        # It returns a list of tuples with (top, right, bottom, left) coordinates for each detected face.
        face_locations = face_recognition.face_locations(frame)
        # Q: What information does face_recognition.face_locations() provide?
        # A: It provides the bounding box coordinates for each face detected in the frame.

        # Loop over each detected face location.
        for face_location in face_locations:
            top, right, bottom, left = face_location
            # Draw a rectangle around the detected face.
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Q: What does cv2.rectangle() do?
            # A: It draws a rectangle on the image using the specified coordinates and color (green in this case).

            # Extract facial landmarks for the detected face.
            # The function face_recognition.face_landmarks() returns a dictionary with keys like "chin", "left_eye", etc.
            landmarks = face_recognition.face_landmarks(frame, [face_location])[0]
            # Q: Why are landmarks useful?
            # A: They provide detailed facial feature points (e.g., eyes, nose, mouth) which can be used for further processing like expression analysis.

            # Loop over each type of landmark and its points.
            for landmark_type, landmark_points in landmarks.items():
                # Draw a small circle at each landmark point.
                for point in landmark_points:
                    cv2.circle(frame, point, 2, (0, 0, 255), -1)
                    # Q: What does cv2.circle() do?
                    # A: It draws a circle at the specified point with a given radius and color (red in this case).

        # Display the frame with the face rectangles and landmark circles.
        cv2.imshow("frame", frame)

        # Exit the loop if the 'q' key is pressed.
        if cv2.waitKey(10) == ord('q'):
            break

    except KeyboardInterrupt:
        # Gracefully exit on keyboard interrupt (e.g., Ctrl+C).
        break

# Release the video capture object and close any OpenCV windows.
cap.release()
cv2.destroyAllWindows()
# Q: Why is it important to release the video capture and destroy windows?
# A: To free up system resources and ensure that the program terminates cleanly.
