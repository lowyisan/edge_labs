#!/usr/bin/env python3
"""
Top-Level Explanation:
This script performs real-time color detection using OpenCV. It captures video from the webcam,
segments the image based on predefined BGR color boundaries (for red, blue, and green),
normalizes the segmented images, and displays the original frame alongside the segmented color outputs.
This demo is useful for understanding color space manipulation, masking, and image normalization in OpenCV,
and it could be useful for lab tests on image processing and computer vision.
Reference: https://pyimagesearch.com/2014/08/04/opencv-python-color-detection/
"""

#%% Import necessary libraries
import cv2              # OpenCV library for image and video processing.
                        # Q: Why use OpenCV?
                        # A: It provides powerful tools for real-time computer vision tasks.
import numpy as np      # For numerical operations and array manipulation.

#%% Define color boundaries in the BGR color space
# OpenCV uses BGR ordering for images (not RGB), so our boundaries are specified accordingly.
# Each boundary is defined as a tuple of two lists: [lower_bound, upper_bound].
# These boundaries are used to detect colors in the image via thresholding.
boundaries = [
    ([17, 15, 100], [50, 56, 200]),  # For Red
    ([86, 31, 4], [220, 88, 50]),    # For Blue
    ([25, 90, 4], [62, 200, 50])     # For Green
]
# Q: Why are the boundaries specified in BGR order?
# A: Because OpenCV represents images in BGR, so the channels are ordered Blue, Green, Red.

#%% Normalize the Image for Display (Optional)
def normalizeImg(Img):
    """
    Normalize an image to the 0-255 range.

    Converts the image to float to perform division without errors, then scales
    the pixel values to span from 0 to 255, and finally converts the result back to uint8.
    
    Q: Why is normalization necessary?
    A: It improves image visualization by adjusting the intensity values to the full 0-255 range.
    """
    Img = np.float64(Img)  # Convert to float to avoid integer division issues.
    norm_img = (Img - np.min(Img)) / (np.max(Img) - np.min(Img))  # Normalize to [0, 1].
    norm_img = np.uint8(norm_img * 255.0)  # Scale to [0, 255] and convert to 8-bit unsigned integer.
    return norm_img

#%% OpenCV Video Capture and Frame Analysis
cap = cv2.VideoCapture(0)  # Open the default webcam (index 0).
# Q: What does cv2.VideoCapture(0) do?
# A: It initializes video capturing from the default camera.

# Check if the webcam is opened correctly; if not, raise an error.
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#%% Process video frames in a loop until 'q' is pressed or interrupted.
while True:
    try:
        # Capture one frame from the webcam.
        ret, frame = cap.read()    
        # Q: What does cap.read() return?
        # A: It returns a boolean indicating success and the captured frame.

        output = []  # Initialize a list to store segmented images for each color boundary.
        
        # Loop over each defined color boundary.
        for (lower, upper) in boundaries:
            # Create NumPy arrays for the lower and upper bounds.
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            
            # Create a mask that identifies pixels within the specified boundary.
            mask = cv2.inRange(frame, lower, upper)
            # Q: What does cv2.inRange() do?
            # A: It thresholds the image, creating a binary mask where pixels falling within the range are set to 255 (white) and others to 0 (black).
            
            # Apply the mask to the frame to segment out the desired color.
            segmented = cv2.bitwise_and(frame, frame, mask=mask)
            output.append(segmented)  # Append the segmented image to the output list.

        # Normalize the segmented images for display.
        # Note: The order in the output list corresponds to the boundaries order.
        red_img = normalizeImg(output[0])
        green_img = normalizeImg(output[1])
        blue_img = normalizeImg(output[2])
        # Q: Why normalize the segmented images?
        # A: Normalization improves visualization by scaling the pixel intensities consistently.
       
        # Concatenate the original frame and the segmented images horizontally for display.
        catImg = cv2.hconcat([frame, red_img, green_img, blue_img])
        
        # Display the concatenated image.
        cv2.imshow("Images with Colours", catImg)
        
        # Wait for 1 ms for a key press; if 'q' is pressed, break the loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
    except KeyboardInterrupt:
        # Allow graceful exit if the user interrupts the program.
        break

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
# Q: Why release the video capture and destroy windows?
# A: To free system resources and close display windows properly after the program ends.
