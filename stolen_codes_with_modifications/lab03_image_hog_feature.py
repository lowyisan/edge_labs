#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-Level Explanation:
This script captures video from the webcam and performs real-time Histogram of Oriented Gradients (HOG)
feature extraction using scikit-image. It converts the frames to grayscale (required by HOG),
computes HOG features along with a visualization image, enhances the visualization, and then
concatenates the original frame with the HOG visualization for side-by-side display.
This demonstration is useful for lab tests on feature extraction, image processing, and real-time
computer vision applications.

Potential Lab Test Questions:
Q1. What is HOG and why is it used?
   A1. HOG (Histogram of Oriented Gradients) is a feature descriptor that captures edge and gradient structures,
       often used for object detection and image analysis.
Q2. Why do we convert the image to grayscale before computing HOG features?
   A2. HOG works on intensity gradients and is designed for single-channel images; converting to grayscale simplifies the
       computation and focuses on shape rather than color.
Q3. How does exposure.rescale_intensity help in visualizing the HOG image?
   A3. It rescales the HOG visualization to the 0–255 range, enhancing contrast and making it suitable for display.
"""

#%% Import required libraries
import cv2                    # OpenCV for video capture and image processing
import numpy as np            # NumPy for numerical operations
from skimage import feature   # For HOG feature extraction
from skimage import exposure  # For enhancing/normalizing image intensity

#%% OpenCV Webcam Capture
# Start video capture from the default webcam (device index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is available
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#%% Main loop to process webcam frames
# This loop will continue until the 'q' key is pressed
while True:
    try:
        # Read a single frame from the webcam
        ret, frame = cap.read()

        # OPTIONAL: Resize the image to 256x256 for faster processing
        # frame = cv2.resize(frame, (256, 256))

        # Convert the frame to grayscale (required for HOG feature extraction)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === HOG Feature Extraction ===
        # Compute HOG features and obtain a visualization of the gradient image.
        (H, hogImage) = feature.hog(image,
                                    orientations=9,              # Number of gradient bins
                                    pixels_per_cell=(8, 8),      # Size of each cell
                                    cells_per_block=(2, 2),      # Group of cells for block normalization
                                    transform_sqrt=True,         # Apply power law compression to normalize contrast
                                    block_norm="L1",             # Block normalization method (L1 norm)
                                    visualize=True)              # Return a HOG image for visualization

# For pixels_per_cell:
# Increasing pixels_per_cell (e.g., from (8, 8) to (16, 16)) averages gradients over a larger area,
# resulting in a coarser, lower-dimensional feature representation with less fine detail.
# Decreasing pixels_per_cell (e.g., from (8, 8) to (4, 4)) captures more detailed gradients,
# yielding a higher-dimensional feature vector that is more sensitive to noise.

# For cells_per_block:
# Increasing cells_per_block (e.g., from (2, 2) to (3, 3)) enlarges the area for normalization,
# making the descriptor more robust to local contrast variations but potentially smoothing out subtle features.
# Decreasing cells_per_block reduces the normalization region, preserving local variations
# but might make the descriptor more vulnerable to noise.


        # === Enhance HOG Visualization ===
        # Rescale the intensity of the HOG visualization to enhance contrast
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")

        # Convert the HOG visualization image to a 3-channel RGB image for concatenation with the original frame
        hogImg = cv2.cvtColor(hogImage, cv2.COLOR_GRAY2RGB)

        # Concatenate the original frame and the HOG visualization image side by side
        catImg = cv2.hconcat([frame, hogImg])

        # Display the concatenated image in a window
        cv2.imshow("HOG Image", catImg)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break  # Allow graceful exit with Ctrl+C

#%% Cleanup
# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
