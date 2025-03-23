#!/usr/bin/env python3
"""
Top-Level Explanation:
This script demonstrates real-time image classification using a pre-trained MobileNet_V2 model from PyTorch.
It captures frames from a webcam using OpenCV, preprocesses the images to match the model's input requirements,
and performs inference to classify the image. The code can run either in a standard or quantized mode for efficiency.
It logs the model's performance in frames per second (fps). This demo is useful for understanding neural network
inference, model quantization, and real-time computer vision applications in Python.
"""

import time  # For timing and performance measurement.

import torch  # PyTorch library for deep learning.
import numpy as np  # NumPy for numerical operations and array handling.
from torchvision import models, transforms  # Pre-trained models and image transforms.
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights  # Quantized weights for MobileNet V2.

import cv2  # OpenCV for video capture and image processing.
from PIL import Image  # PIL for image handling if needed (not used directly in this script).

# Flag to decide whether to run in quantized mode or not.
quantize = False

# If quantization is enabled, set the quantization engine.
if quantize:
    torch.backends.quantized.engine = 'qnnpack'
    # Q: What is quantization in neural networks?
    # A: Quantization reduces the precision of the weights and activations, leading to faster inference and reduced memory usage.

# Open the default webcam.
cap = cv2.VideoCapture(0)
# Set the frame dimensions to 224x224 pixels, which is the standard input size for MobileNet.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
# Set the desired frames per second.
cap.set(cv2.CAP_PROP_FPS, 36)
# Q: Why is it important to set the frame size to 224x224?
# A: The MobileNet_V2 model expects 224x224 input images, so we need to match this dimension for proper inference.

# Define the preprocessing transformations.
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet's mean...
                         std=[0.229, 0.224, 0.225]),   # ...and standard deviation.
])
# Q: Why normalize the image?
# A: Normalization scales the pixel values to a standard range, matching the conditions the model was trained under.

# Load the default quantized weights for MobileNet_V2 if quantization is enabled.
weights = MobileNet_V2_QuantizedWeights.DEFAULT
classes = weights.meta["categories"]  # Retrieve class names from the model metadata.

# Load the pre-trained MobileNet_V2 model; use quantization if specified.
net = models.quantization.mobilenet_v2(pretrained=True, quantize=quantize)
# Q: What is MobileNet_V2?
# A: MobileNet_V2 is a lightweight convolutional neural network optimized for mobile and embedded vision applications.

# Variables for performance measurement.
started = time.time()
last_logged = time.time()
frame_count = 0

# Disable gradient computation for inference (saves memory and computation).
with torch.no_grad():
    while True:
        # Read a frame from the webcam.
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")
        
        # Convert the OpenCV image from BGR to RGB.
        image = image[:, :, [2, 1, 0]]
        # Q: Why convert from BGR to RGB?
        # A: OpenCV uses BGR by default, but the model expects images in RGB format.

        # Preprocess the image: convert to tensor and normalize.
        input_tensor = preprocess(image)
        
        # Create a mini-batch (add batch dimension) as expected by the model.
        input_batch = input_tensor.unsqueeze(0)
        
        # Run the model on the input batch.
        output = net(input_batch)
        
        # Uncomment the following lines to print top 10 predictions:
        # top = list(enumerate(output[0].softmax(dim=0)))
        # top.sort(key=lambda x: x[1], reverse=True)
        # for idx, val in top[:10]:
        #     print(f"{val.item()*100:.2f}% {classes[idx]}")
        # print(f"========================================================================")
        # Q: What does softmax do in this context?
        # A: Softmax converts the raw model outputs (logits) into probabilities, making it easier to interpret predictions.

        # Log the model's performance by calculating frames per second (fps).
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"============={frame_count / (now - last_logged)} fps =================")
            last_logged = now
            frame_count = 0

        # Optional: Add code here to display the frame or further process the output.
        # For example, use cv2.imshow("Frame", image) if you wish to see the live video.

        # Exit condition could be added here if integrating with cv2.imshow (e.g., if cv2.waitKey(1) == ord('q'): break)
