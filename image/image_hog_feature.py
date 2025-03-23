#!/usr/bin/env python3
"""
Top-Level Explanation:
This script captures real-time video from a webcam, extracts Histogram of Oriented Gradients (HOG) features
from each frame using the scikit-image library, and displays both the original frame and its corresponding
HOG visualization side by side. HOG features are useful for object detection and image analysis tasks.
This demo is useful for understanding feature extraction in computer vision and can help answer questions
related to image processing techniques, performance considerations, and working with video streams.
Reference: https://scikit-image.org/
"""

import cv2                     # OpenCV library for video capture and image processing.
import numpy as np             # NumPy for efficient array operations.
from skimage import feature   # Provides the HOG function for feature extraction.
from skimage import exposure  # Used for image intensity rescaling (contrast stretching).

#%% OpenCV Video Capture and Frame Analysis
cap = cv2.VideoCapture(0)  # Open the default webcam (index 0).
# Q: What does cv2.VideoCapture(0) do?
# A: It initializes video capture from the default camera.

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")  # Raise an error if the webcam cannot be accessed.

# The loop will break on pressing the 'q' key
while True:
    try:
        # Capture one frame from the webcam.
        ret, frame = cap.read()  
        # Q: What do ret and frame represent?
        # A: 'ret' is a boolean indicating success, and 'frame' is the captured image.

        # Optional: Resize the frame for faster processing.
        # Uncomment the following line to resize the frame to 256x256 pixels.
        # frame = cv2.resize(frame, (256, 256))
        # Q: How does resizing affect processing speed?
        # A: Smaller images require less computation, leading to faster processing times.

        # Convert the captured frame to grayscale.
        # HOG feature extraction in scikit-image works only on grayscale images.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Q: Why convert the image to grayscale?
        # A: To reduce computational complexity and because HOG typically operates on single channel images.

        # Extract the HOG features from the grayscale image.
        # 'H' contains the HOG feature vector; 'hogImage' is a visualization of the HOG.
        (H, hogImage) = feature.hog(image,
                                    orientations=9,         # Number of orientation bins.
                                    pixels_per_cell=(8, 8),   # Size (in pixels) of a cell.
                                    cells_per_block=(2, 2),   # Number of cells in each block.
                                    transform_sqrt=True,      # Apply power law compression.
                                    block_norm="L1",          # Block normalization method.
                                    visualize=True)           # Return the HOG image for visualization.
        # Q: What is the purpose of the 'visualize' parameter?
        # A: It returns an image representation of the HOG, which helps in understanding the gradient distribution.

        # Rescale the intensity of the HOG visualization to the range 0-255.
        # This is done to improve the contrast of the HOG image.
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")  # Convert the image to 8-bit unsigned integers.
        # Q: Why rescale intensity?
        # A: To adjust the image contrast for better visual interpretation.

        # Convert the grayscale HOG image back to RGB for concatenation with the original frame.
        hogImg = cv2.cvtColor(hogImage, cv2.COLOR_GRAY2RGB)
        # Q: Why convert from grayscale to RGB?
        # A: To match the color channels of the original frame when concatenating for display.

        # Concatenate the original frame and the HOG visualization horizontally.
        catImg = cv2.hconcat([frame, hogImg])
        # Q: What does cv2.hconcat() do?
        # A: It combines two images horizontally (side-by-side).

        # Display the concatenated image in a window titled "HOG Image".
        cv2.imshow("HOG Image", catImg)
        
        # Exit the loop when the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except KeyboardInterrupt:
        # Allow graceful exit if a keyboard interrupt (Ctrl+C) occurs.
        break

# Release the video capture object and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
# Q: Why is it important to release the capture and destroy windows?
# A: To free system resources and ensure the program terminates cleanly.
