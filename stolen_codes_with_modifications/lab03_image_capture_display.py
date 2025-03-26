#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script performs real-time color detection using OpenCV. It captures video from the webcam,
applies color segmentation based on defined BGR color boundaries, normalizes the segmented outputs,
and displays both the original and segmented images side by side. This demonstration is useful for
understanding color space manipulation, image masking, and normalization in OpenCV, and can be applied
in lab tests on computer vision and image processing.

Potential Lab Test Questions:
Q1. Why are the color boundaries defined in BGR instead of RGB?
   A1. OpenCV represents images in BGR format by default, so the boundaries must be set accordingly.
Q2. What is the purpose of the normalizeImg function?
   A2. It normalizes image pixel values to the 0–255 range, enhancing the contrast and visibility of the segmented outputs.
Q3. How does cv2.inRange help in color segmentation?
   A3. It creates a binary mask that identifies pixels within the specified color range, which can then be used to extract those regions.
"""

#%% Import required libraries
import cv2                  # OpenCV for computer vision and webcam capture
import numpy as np          # NumPy for numerical operations and masking

#%% Define BGR color boundaries for segmentation
# The boundaries are defined for detecting Red, Blue, and Green colors.
# Note: Although comments mention "RGB", OpenCV uses BGR format by default.
# A trick to determine these ranges for a new colour: think of each channel (B, G, R) as a slider.
# For example, if you want to detect a color X that has high Blue and Green but low Red, set high ranges for Blue and Green,
# and a lower range for Red.
boundaries = [
    ([17, 15, 100], [50, 56, 200]),   # Red range (BGR values)
    ([86, 31, 4], [220, 88, 50]),      # Blue range (BGR values)
    ([25, 90, 4], [62, 200, 50]),      # Green range (BGR values)
    ([0, 100, 100], [50, 255, 255]),   # Yellow range (BGR values)
    ([0, 100, 200], [50, 200, 255]),   # Orange range (BGR values)
    ([100, 0, 100], [160, 100, 160]),  # Purple range (BGR values)
    ([200, 200, 0], [255, 255, 50]),   # Cyan range (BGR values)
    ([200, 0, 200], [255, 100, 255]),  # Magenta range (BGR values)
    ([20, 20, 140], [70, 70, 200])     # Brown range (BGR values)
]

#%% Utility function to normalize images for display
def normalizeImg(Img):
    """
    Normalize the input image to the range 0–255.
    This helps in enhancing the contrast of the masked output.

    Parameters:
        Img: Input image (NumPy array)

    Returns:
        norm_img: Normalized image in 8-bit format
    """
    Img = np.float64(Img)  # Convert to float for safe division
    # Normalize to [0,1] by subtracting the min and dividing by the range.
    # Added 1e-6 to avoid division by zero.
    norm_img = (Img - np.min(Img)) / (np.max(Img) - np.min(Img) + 1e-6)
    norm_img = np.uint8(norm_img * 255.0)  # Scale to 8-bit range (0-255)
    return norm_img

#%% Start capturing video from the webcam
cap = cv2.VideoCapture(0)  # Open the default webcam (device 0)

# Verify that the webcam opened successfully; if not, raise an error.
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#%% Main loop for frame capture and processing
# Press 'q' to exit the loop.
while True:
    try:
        # Read a frame from the webcam.
        ret, frame = cap.read()
        # ret is a boolean indicating if the frame was captured successfully.
        
        output = []  # List to store segmented images for each defined color

        # Loop over each defined BGR color boundary.
        for (lower, upper) in boundaries:
            # Convert lower and upper boundary lists to NumPy arrays.
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            
            # Create a binary mask where pixels within the boundary are set to 255 (white).
            mask = cv2.inRange(frame, lower, upper)
            # Q: What does cv2.inRange do?
            # A: It thresholds the image, setting pixels within the given range to white and others to black.
            
            # Apply the mask to the original frame to extract the color-specific regions.
            segmented = cv2.bitwise_and(frame, frame, mask=mask)
            output.append(segmented)

        # Normalize each segmented image for better visualization.
        red_img = normalizeImg(output[0])
        green_img = normalizeImg(output[1])
        blue_img = normalizeImg(output[2])
        # Q: Why normalize the segmented images?
        # A: Normalization scales the pixel intensities uniformly, improving contrast for display.

        # Concatenate the original frame and the segmented images horizontally.
        catImg = cv2.hconcat([frame, red_img, green_img, blue_img])
        
        # Display the concatenated image.
        cv2.imshow("Images with Colours", catImg)

        # Exit the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Allow graceful exit if the user interrupts the program.
        break

#%% Cleanup after loop exit
cap.release()             # Release the webcam resource
cv2.destroyAllWindows()   # Close all OpenCV windows
