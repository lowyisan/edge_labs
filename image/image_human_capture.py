#!/usr/bin/env python3
"""
Top-Level Explanation:
This script performs real-time person detection using OpenCV's HOG (Histogram of Oriented Gradients)
descriptor combined with a pre-trained SVM detector for people. It captures video from the webcam,
detects people in each frame, calculates the horizontal distance of each detected person’s bounding box
from a fixed center, and then prints commands (e.g., "left", "right", "center") based on that distance.
This demo can be used to learn about object detection, bounding box processing, and basic decision logic
in computer vision applications, which may be relevant for your lab test.
"""

import cv2  # OpenCV for video capture and image processing.
import numpy as np  # NumPy for efficient numerical and array operations.

#%% Initialize the HOG descriptor/person detector
# Create an instance of the HOGDescriptor.
hog = cv2.HOGDescriptor()
# Set the SVM detector with a pre-trained people detector.
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# Q: Why do we use HOG for person detection?
# A: HOG features are effective for capturing the shape and appearance of people in images.

# Define a tolerance for center alignment.
# This value determines how many pixels away from the center a person can be
# before the system decides the person is off-center.
center_tolerance = 5

#%% OpenCV Video Capture and Frame Analysis
cap = cv2.VideoCapture(0)  # Open the default webcam (index 0).
# Q: What does cv2.VideoCapture(0) do?
# A: It initializes video capture from the default camera.

# Check if the webcam is opened correctly.
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Process frames continuously until 'q' is pressed.
while True:
    try:
        # Capture one frame from the webcam.
        ret, frame = cap.read()
        # Q: What are 'ret' and 'frame'?
        # A: 'ret' is a boolean indicating if the frame was successfully captured, and 'frame' is the image.

        # Resize frame for faster detection.
        frame = cv2.resize(frame, (256, 256))
        # Q: How does resizing affect performance?
        # A: Smaller frames require less computation, speeding up detection at the cost of resolution.

        # Detect people in the image.
        # 'detectMultiScale' returns bounding boxes and weights for detected objects.
        boxes, weights = hog.detectMultiScale(frame, winStride=(1, 1), scale=1.05)
        # Convert bounding boxes to format: [x1, y1, x2, y2].
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        centers = []  # List to store each box and its distance from a defined center.

        # Loop over detected bounding boxes to compute horizontal center position.
        for box in boxes:
            # Calculate center x-coordinate of the bounding box.
            center_x = ((box[2] - box[0]) / 2) + box[0]
            # Compute horizontal distance from a fixed center (70 in this case).
            x_pos_rel_center = center_x - 70
            # Get absolute distance.
            dist_to_center_x = abs(x_pos_rel_center)
            # Append the box information and calculated values.
            centers.append({
                'box': box,
                'x_pos_rel_center': x_pos_rel_center,
                'dist_to_center_x': dist_to_center_x
            })
        # Q: What is the purpose of computing the distance from the center?
        # A: It determines whether the detected person is centered, to the left, or right relative to a reference point.

        if len(centers) > 0:
            # Sort the detected boxes by distance to the center.
            sorted_boxes = sorted(centers, key=lambda i: i['dist_to_center_x'])
            # Draw the boxes: the most centered person is highlighted differently.
            for idx in range(len(sorted_boxes)):
                box = sorted_boxes[idx]['box']
                # Draw a green rectangle for the most centered box.
                if idx == 0:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                else:
                    # Draw red rectangles for the other boxes.
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # Retrieve the horizontal offset of the most centered box.
            Center_box_pos_x = sorted_boxes[0]['x_pos_rel_center']
            # Determine position relative to center and print corresponding message.
            if -center_tolerance <= Center_box_pos_x <= center_tolerance:
                # Person is centered; simulate turning on eye light.
                print("center")
            elif Center_box_pos_x >= center_tolerance:
                # Person is to the right; instruct head turn to the right.
                print("right")
            elif Center_box_pos_x <= -center_tolerance:
                # Person is to the left; instruct head turn to the left.
                print("left")
            # Print the calculated offset value.
            print(str(Center_box_pos_x))
        else:
            # No person detected in the frame.
            print("nothing detected")

        # Resize the frame for display purposes (easier to view on screen).
        frame = cv2.resize(frame, (720, 720))
        # Display the resulting frame.
        cv2.imshow("frame", frame)

        # Exit the loop if 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Gracefully exit on keyboard interrupt (e.g., Ctrl+C).
        break

# Release the video capture object and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
# Q: Why is it important to release the video capture and destroy windows?
# A: To free system resources and ensure that the program terminates cleanly.
