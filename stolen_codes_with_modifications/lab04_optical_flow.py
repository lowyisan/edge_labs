#%% Top-Level Explanation
# This script demonstrates real-time optical flow estimation and tracking using OpenCV.
# It implements two methods for optical flow:
#   1. Sparse Optical Flow using Lucas-Kanade (tracking a set of detected feature points).
#   2. Dense Optical Flow using Farneback (estimating flow for every pixel on a sampled grid).
# The script processes video frames from a webcam to visualize motion vectors, which indicate
# the direction and magnitude of movement between frames.
#
# Potential Lab Test Q&A:
# Q: What is optical flow?
# A: Optical flow is the pattern of apparent motion of objects in a visual scene caused by the relative motion
#    between an observer and the scene.
#
# Q: What are the differences between sparse and dense optical flow?
# A: Sparse optical flow tracks selected key feature points (e.g., using Lucas-Kanade), while dense optical flow
#    computes motion for every pixel or a grid of pixels (e.g., using Farneback).
#
# Q: How does the Lucas-Kanade method work in this script?
# A: It detects strong features using Shi-Tomasi corner detection, then tracks these points across frames using
#    the Lucas-Kanade method. Motion vectors are drawn from previous positions to current positions.
#
# Q: Why might one choose dense optical flow over sparse optical flow?
# A: Dense optical flow provides a more complete picture of the motion field over the entire image, which is useful
#    for applications that require a detailed analysis of movement.

import numpy as np         # For numerical operations and handling arrays
import cv2                 # OpenCV for video capture, image processing, and drawing
# No need for time import here, but it's useful for potential timestamping if needed

#%% Generic Parameters
# Create an array of 100 random RGB colors for visually differentiating motion vectors.
# Increasing this value allows more unique colors for more tracking points.
color = np.random.randint(0, 255, (100, 3))

#%% Parameters for Lucas-Kanade Optical Flow (Sparse Tracking)
# These parameters are used with the Shi-Tomasi method to detect strong feature points in the first frame.
feature_params = dict(
    maxCorners=100,       # Maximum number of corners to detect; more corners provide more detail but require more CPU.
    qualityLevel=0.3,     # Minimum quality of corners (0 to 1); lower value includes more corners but may include noise.
    minDistance=7,        # Minimum distance in pixels between detected corners; smaller values yield denser features.
    blockSize=7           # Size of the averaging block used in corner detection; larger values produce smoother results.
)

# Parameters for Lucas-Kanade optical flow algorithm (cv2.calcOpticalFlowPyrLK)
lk_params = dict(
    winSize=(15, 15),     # Size of the search window at each pyramid level; larger windows handle bigger motions but may lose detail.
    maxLevel=2,           # Number of pyramid levels used; higher values help capture larger motions at the cost of speed.
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,               # Maximum number of iterations for refining the flow estimation.
        0.03              # Minimum error threshold for convergence; lower value means more accurate but slower.
    )
)

#%% Setup function for first frame
def set1stFrame(frame):
    """
    Prepares the first frame for optical flow estimation.
    - Converts the frame to grayscale.
    - Detects strong feature points using the Shi-Tomasi method.
    - Creates an empty mask for drawing motion vectors.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)  # Detect feature points
    mask = np.zeros_like(frame)  # Create a black mask for drawing lines
    return frame_gray, mask, p0

#%% Lucas-Kanade Optical Flow (Sparse)
def LucasKanadeOpticalFlow(frame, old_gray, mask, p0):
    """
    Computes sparse optical flow using Lucas-Kanade method.
    - Converts the current frame to grayscale.
    - Uses calcOpticalFlowPyrLK to compute the new positions of feature points.
    - Filters out points that were not successfully tracked.
    - Draws motion vectors (lines) and circles on the new feature points.
    - Updates the previous frame and feature points for the next iteration.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale

    # Fallback in case no points are available
    if (p0 is None or len(p0) == 0):
        p0 = np.array([[50, 50], [100, 100]], dtype=np.float32).reshape(-1, 1, 2)

    # Calculate optical flow between old and current frames
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    if p1 is not None:
        # Select only the successfully tracked points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
        # Draw motion vectors for each tracked point
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()  # New point coordinates
            c, d = old.ravel()  # Old point coordinates
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)  # Draw line
            frame_gray = cv2.circle(frame_gray, (int(a), int(b)), 5, color[i].tolist(), -1)  # Draw circle

        img = cv2.add(frame, mask)  # Overlay the mask on the current frame
        old_gray = frame_gray.copy()  # Update the old frame for the next iteration
        p0 = good_new.reshape(-1, 1, 2)  # Update the feature points

    return img, old_gray, p0

#%% Farneback Dense Optical Flow
# Parameters for dense optical flow visualization
step = 16  # Sampling step for drawing flow vectors; lower values give denser visualization, higher values for speed

def DenseOpticalFlowByLines(frame, old_gray):
    """
    Computes dense optical flow using the Farneback method.
    - Converts the current frame to grayscale.
    - Computes the optical flow field.
    - Samples a grid of points and calculates flow vectors at those points.
    - Draws lines and circles to represent the flow vectors.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale
    h, w = frame_gray.shape[:2]  # Get frame dimensions

    # Create a grid of points for which to visualize the flow
    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1)
    
    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        old_gray, frame_gray, None,
        pyr_scale=0.5,  # Scaling between pyramid levels; lower values capture more detail
        levels=3,       # Number of pyramid layers
        winsize=15,     # Averaging window size
        iterations=3,   # Number of iterations per pyramid level
        poly_n=5,       # Size of the pixel neighborhood used for polynomial expansion
        poly_sigma=1.2, # Standard deviation for Gaussian smoothing
        flags=0
    )

    # Obtain flow vectors (fx, fy) at the sampled grid points
    fx, fy = flow[y, x].T

    # Create line endpoints: starting point (x, y) and endpoint (x+fx, y+fy)
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # Draw lines representing motion vectors on the frame
    cv2.polylines(frame, lines, isClosed=False, color=(0, 255, 0))

    # Optionally, draw a small circle at the origin of each vector
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)

    return frame

#%% Initialize Video Capture
cap = cv2.VideoCapture(0)  # Start capturing video from the default webcam (device 0)

# Ensure that the webcam is accessible
if not cap.isOpened():
    raise IOError("Cannot open webcam")

firstframeflag = 1  # Flag to indicate that the first frame has not yet been processed

# Main loop: process frames from the webcam continuously
while True:
    try:
        if firstframeflag:
            # Read the first frame to initialize optical flow parameters
            ret, frame = cap.read()
            # Setup initial grayscale frame, feature points, and drawing mask
            old_gray, mask, p0 = set1stFrame(frame)
            firstframeflag = 0  # Reset the flag after processing the first frame

        # Read the next frame from the webcam
        ret, frame = cap.read()

        # === USE DENSE OPTICAL FLOW ===
        # Compute dense optical flow and draw motion vectors as lines
        img = DenseOpticalFlowByLines(frame, old_gray)

        # === TO USE SPARSE TRACKING (LUCAS-KANADE), UNCOMMENT BELOW ===
        # img, old_gray, p0 = LucasKanadeOpticalFlow(frame, old_gray, mask, p0)

        # Display the processed frame with motion vectors
        cv2.imshow("Optical Flow", img)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Graceful exit when interrupted with Ctrl+C
        break

# Cleanup: release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
