#!/usr/bin/env python3
"""
Top-Level Explanation:
This script performs real-time optical flow estimation and tracking using OpenCV.
It demonstrates two optical flow techniques:
1. Sparse optical flow using the Lucas-Kanade method, which tracks specific feature points.
2. Dense optical flow using Gunnar Farneback’s algorithm, which computes flow vectors for a grid of pixels.
The script captures video from a webcam, processes each frame to estimate motion between consecutive frames,
and visualizes the motion either as drawn tracks (for Lucas-Kanade) or as flow lines (for Farneback).
This code is useful for understanding motion analysis and tracking in video sequences, which are common topics in computer vision labs.
Reference: https://github.com/daisukelab/cv_opt_flow/tree/master
"""

import numpy as np            # NumPy for numerical operations and array manipulation.
import cv2                    # OpenCV for image processing, video capture, and optical flow estimation.

#%% Generic Parameters
# Generate an array of 100 random colors for visualizing different feature tracks.
color = np.random.randint(0, 255, (100, 3))  # Each row is a random BGR color.
# Q: Why use random colors?
# A: To easily distinguish different feature points or flow vectors when drawn on the frame.

#%% Parameters for Lucas-Kanade Optical Flow Approach
# Parameters for Shi-Tomasi corner detection (good features to track).
feature_params = dict(
    maxCorners=100,       # Maximum number of corners to return.
    qualityLevel=0.3,     # Minimal accepted quality of image corners.
    minDistance=7,        # Minimum possible Euclidean distance between returned corners.
    blockSize=7           # Size of an average block for computing a derivative covariance matrix.
)
# Parameters for Lucas-Kanade optical flow.
lk_params = dict(
    winSize=(15, 15),     # Size of the search window at each pyramid level.
    maxLevel=2,           # Maximum number of pyramid levels.
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Termination criteria.
)
# Q: What is the Lucas-Kanade method used for?
# A: It estimates motion by tracking sparse feature points between consecutive frames.

#%% Function: Initialize First Frame for Optical Flow
def set1stFrame(frame):
    """
    Prepares the first frame for optical flow estimation.
    Converts the frame to grayscale, detects good features to track (corners),
    and creates a mask for drawing the optical flow trajectories.

    Parameters:
      frame: The initial video frame (color image).

    Returns:
      frame_gray: The grayscale version of the frame.
      mask: An empty mask image with the same size as the frame for drawing.
      p0: Detected feature points (corners) for tracking.
    """
    # Convert the input frame to grayscale.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect good features to track using Shi-Tomasi method.
    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
    # Create a blank mask image (same dimensions as frame) to draw optical flow paths.
    mask = np.zeros_like(frame)
    return frame_gray, mask, p0

#%% Function: Lucas-Kanade Optical Flow Tracking
def LucasKanadeOpticalFlow(frame, old_gray, mask, p0):
    """
    Tracks feature points using the Lucas-Kanade optical flow method and draws their trajectories.

    Parameters:
      frame: The current video frame (color image).
      old_gray: The previous frame in grayscale.
      mask: A mask image used for drawing optical flow lines.
      p0: Feature points from the previous frame.

    Returns:
      img: The current frame with drawn optical flow paths.
      old_gray: The updated previous frame (current frame in grayscale).
      p0: Updated feature points for tracking in the next frame.
    """
    # Convert the current frame to grayscale.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Check if there are valid feature points; if not, create a default set.
    if (p0 is None or len(p0) == 0):
        p0 = np.array([[50, 50], [100, 100]], dtype=np.float32).reshape(-1, 1, 2)
    
    # Calculate optical flow: find new positions of the previously tracked points.
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    if p1 is not None:
        # Select good points that were successfully tracked.
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
        # Draw the optical flow tracks.
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()  # New coordinates.
            c, d = old.ravel()  # Old coordinates.
            # Draw a line from the old position to the new position.
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            # Draw a circle at the new position.
            frame_gray = cv2.circle(frame_gray, (int(a), int(b)), 5, color[i].tolist(), -1)
        # Combine the original frame and the mask with drawn tracks.
        img = cv2.add(frame, mask)
    
        # Update old_gray to the current frame for the next iteration.
        old_gray = frame_gray.copy()
        # Update feature points with the new positions.
        p0 = good_new.reshape(-1, 1, 2)
    
    return img, old_gray, p0

#%% Function: Dense Optical Flow using Farneback's Algorithm
step = 16  # Define the grid step for sampling flow vectors.

def DenseOpticalFlowByLines(frame, old_gray):
    """
    Computes dense optical flow using Farneback's algorithm and visualizes it as flow lines.

    Parameters:
      frame: The current video frame (color image).
      old_gray: The previous frame in grayscale.

    Returns:
      frame: The current frame with drawn flow vectors.
    """
    # Convert the current frame to grayscale.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get the height and width of the frame.
    h, w = frame_gray.shape[:2]
    # Create a grid of points at which to sample the optical flow.
    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1)
    
    # Calculate dense optical flow between the previous and current grayscale frames.
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None,
                                          0.5, 3, 15, 3, 5, 1.2, 0)
    # Extract the flow vectors (fx, fy) at the grid points.
    fx, fy = flow[y, x].T
    
    # Prepare lines by combining the starting and ending points of flow vectors.
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # Draw polylines on the original frame representing the flow.
    cv2.polylines(frame, lines, 0, (0, 255, 0))
    # Draw a small circle at the starting point of each flow vector.
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)
    return frame

#%% OpenCV Video Capture and Frame Analysis
cap = cv2.VideoCapture(0)  # Open the default webcam.

# Check if the webcam is opened correctly.
if not cap.isOpened():
    raise IOError("Cannot open webcam")

firstframeflag = 1  # Flag to indicate processing of the first frame.

# Process frames until the 'q' key is pressed.
while True:
    try:
        # For the first frame, initialize the previous frame and feature points.
        if firstframeflag:
            ret, frame = cap.read()  # Capture one frame.
            old_gray, mask, p0 = set1stFrame(frame)
            firstframeflag = 0  # Reset flag after initialization.
        
        # Capture the current frame.
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Option to use Dense Optical Flow visualization.
        img = DenseOpticalFlowByLines(frame, old_gray)
        
        # Option to use Lucas-Kanade sparse optical flow (uncomment the following line to use it).
        # img, old_gray, p0 = LucasKanadeOpticalFlow(frame, old_gray, mask, p0)
        
        # Display the resulting frame with optical flow visualization.
        cv2.imshow("Optical Flow", img)
       
        # Exit the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
    except KeyboardInterrupt:
        # Gracefully exit on keyboard interrupt.
        break

# Release the video capture object and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
# Q: Why is it important to update the previous frame and feature points?
# A: Optical flow is computed relative to the previous frame; updating them ensures that motion is tracked accurately across frames.
