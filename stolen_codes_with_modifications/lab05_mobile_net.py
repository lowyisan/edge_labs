# Top-Level Explanation:
# This script performs real-time image classification using a pre-trained MobileNetV2 model.
# It supports both the floating-point and quantized (INT8) versions of MobileNetV2.
# The code captures images from a webcam, preprocesses them to match the expected input of the model,
# performs inference, and logs the processing speed (FPS). Quantization can be enabled to use the
# more efficient, lower-precision version of the model.
#
# Potential Lab Test Q&A:
# Q: What is quantization and why would you use a quantized model?
# A: Quantization converts a model’s weights and activations from floating point to lower precision (e.g., int8),
#    which reduces the model size and speeds up inference with minimal loss of accuracy.
#
# Q: Why is the image normalized with specific mean and std values?
# A: The normalization values are chosen to match the statistics of the ImageNet dataset, on which MobileNetV2 was trained.
#
# Q: Why do we convert images from BGR to RGB?
# A: OpenCV uses BGR by default, but most deep learning models and torchvision expect images in RGB format.

# Import standard libraries
import time                       # Used for measuring FPS and logging performance

# Import deep learning and image utilities
import torch                      # PyTorch for model loading and inference
import numpy as np                # Numerical operations
from torchvision import models, transforms  # Torchvision models and image preprocessing
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights  # For quantized MobileNetV2

# Import OpenCV and PIL for image capture and handling
import cv2                        # For capturing images from the webcam and image processing
from PIL import Image             # For image conversion and compatibility with torchvision

#%% Configuration

quantize = False  # Set to True if you want to use the quantized (INT8) version of MobileNetV2

# If quantization is enabled, set the quantized engine to 'qnnpack' (required by PyTorch for efficient quantized operations)
if quantize:
    torch.backends.quantized.engine = 'qnnpack'

#%% Initialize Webcam

cap = cv2.VideoCapture(0)  # Open the default camera (device 0)

# Set the frame dimensions to 224x224 — required input size for MobileNetV2
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)  # Optional: set a high FPS for smoother real-time experience

#%% Preprocessing Pipeline

# Define preprocessing steps to convert images to the format expected by MobileNetV2:
# 1. Convert image to tensor
# 2. Normalize using the mean and std of the ImageNet dataset
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert image (H x W x C) in range [0, 255] to a tensor (C x H x W) in range [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225]),  # ImageNet std deviation
])

#%% Load Model and Class Labels

# Load default weights for quantized MobileNetV2 model from the provided metadata
weights = MobileNet_V2_QuantizedWeights.DEFAULT

# Load category labels (ImageNet classes) from the weights metadata
classes = weights.meta["categories"]

# Load the MobileNetV2 model with pretrained weights
# If quantize is True, the model will be loaded as a quantized version (INT8), otherwise as FP32
net = models.quantization.mobilenet_v2(pretrained=True, quantize=quantize)

#%% Performance Tracking

started = time.time()       # Record the start time for total run time measurement
last_logged = time.time()   # Record the last time the FPS was logged
frame_count = 0             # Counter for the number of frames processed since the last FPS log

#%% Inference Loop

with torch.no_grad():  # Disable gradient calculation to speed up inference
    while True:
        # Read a frame from the webcam
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")

        # Convert image from OpenCV's BGR format to RGB format (as expected by PIL and torchvision)
        image = image[:, :, [2, 1, 0]]  # Alternatively, cv2.cvtColor(image, cv2.COLOR_BGR2RGB) can be used

        # (Optional) Conversion to PIL Image can be done if needed:
        # pil_image = Image.fromarray(image)
        # Here we simply rename the variable for clarity
        permuted = image

        # Preprocess the frame: convert to tensor and normalize using the defined preprocessing pipeline
        input_tensor = preprocess(image)

        # Add a batch dimension to create a batch of size 1 (shape: [1, C, H, W])
        input_batch = input_tensor.unsqueeze(0)

        # Run inference using the model on the preprocessed batch
        output = net(input_batch)

        # Uncomment the following block to print the top-10 predictions with confidence values:
        """
        top = list(enumerate(output[0].softmax(dim=0)))
        top.sort(key=lambda x: x[1], reverse=True)
        for idx, val in top[:10]:
            print(f"{val.item()*100:.2f}% {classes[idx]}")
        print(f"========================================================================")
        """

        # Performance logging: calculate frames per second (FPS)
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:  # Log FPS every second
            fps = frame_count / (now - last_logged)
            print(f"============= {fps:.2f} fps =============")
            last_logged = now
            frame_count = 0

# Note:
# The script continuously captures frames, preprocesses them, performs classification, and logs FPS.
# It does not display the classification results on screen, but the commented-out block can be used to
# print the top predictions for debugging or demonstration purposes.
