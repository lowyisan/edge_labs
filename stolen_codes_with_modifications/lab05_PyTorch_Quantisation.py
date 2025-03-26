# Top-Level Explanation:
# This code sets up the environment for training and quantizing a neural network using PyTorch.
# It installs specific versions of torch and torchvision (for compatibility), loads the MNIST dataset,
# and defines helper functions and classes for training (e.g., tracking loss and accuracy) and model quantization.
# The quantization functions (like load_model and fuse_modules) prepare the network for converting from FP32 to INT8,
# which can reduce the model size and speed up inference. This code forms the initial part of a pipeline for
# post-training quantization and quantization-aware training.
#
# Potential Lab Test Q&A:
# Q: Why do we install specific versions of torch and torchvision?
# A: To ensure compatibility with the quantization code and models, as different versions may have different APIs.
#
# Q: What is the purpose of normalization in the transform pipeline?
# A: Normalization scales the pixel values to a standardized range (here, [-1, 1]), which improves the training stability.
#
# Q: How do functions like fuse_modules help with quantization?
# A: They fuse adjacent layers (e.g., Conv + ReLU) to reduce memory access and computational overhead, resulting in
#    a more efficient quantized model.

# Install specific versions of PyTorch and torchvision (required for compatibility)
!pip3 install torch==1.5.0 torchvision==1.6.0

# Import core PyTorch and torchvision modules for model building, datasets, and training
import torch                        # Core PyTorch library for tensor operations and neural network components
import torchvision                  # Contains datasets, models, and transforms for computer vision
import torchvision.transforms as transforms  # For preprocessing and data augmentation
import torch.nn as nn               # For building neural network layers and modules
import torch.nn.functional as F     # For common neural network functions (e.g., activation functions)
import torch.optim as optim         # For optimization algorithms (e.g., SGD, Adam)
import os                           # For interacting with the file system (e.g., saving models)
from torch.utils.data import DataLoader  # For loading and batching datasets
import torch.quantization           # Provides tools for post-training quantization
from torch.quantization import QuantStub, DeQuantStub  # For marking quantization and dequantization points in the model

# Define a transform pipeline to convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),                   # Converts images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))       # Normalizes tensor values to have mean=0.5 and std=0.5 (scales to [-1, 1])
])

# Load MNIST dataset for training with transformations applied
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64,
                         shuffle=True, num_workers=16, pin_memory=True)

# Load MNIST test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64,
                        shuffle=False, num_workers=16, pin_memory=True)

# Utility class for tracking average values (e.g., loss, accuracy) during training
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()  # Initialize or reset the internal state

    def reset(self):
        """Resets all statistics to zero"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the meter with a new value 'val' and count 'n'"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Returns a formatted string representing the current and average values"""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# Compute Top-1 classification accuracy for a batch of outputs
def accuracy(output, target):
    """Computes the top-1 accuracy"""
    with torch.no_grad():  # Disable gradient computation for evaluation
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)  # Get the index of the highest log-probability
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_one = correct[:1].view(-1).float().sum(0, keepdim=True)
        return correct_one.mul_(100.0 / batch_size).item()  # Return accuracy percentage

# Utility function to measure model file size (by temporarily saving state_dict)
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")  # Save model weights to a temporary file
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)  # Print the file size in megabytes
    os.remove("temp.p")  # Remove the temporary file

# Transfers weights from a regular (FP32) model to a quantized model structure
def load_model(quantized_model, model):
    state_dict = model.state_dict()  # Get the state dictionary from the FP32 model
    model = model.to('cpu')          # Ensure the model is on CPU
    quantized_model.load_state_dict(state_dict)  # Load the weights into the quantized model

# Fuses layers such as Conv+ReLU or Linear+ReLU to optimize for quantization.
# Fusing reduces memory accesses and can improve both speed and accuracy of quantized models.
def fuse_modules(model):
    torch.quantization.fuse_modules(model, [['conv1', 'relu1'],
                                            ['conv2', 'relu2'],
                                            ['fc1', 'relu3'],
                                            ['fc2', 'relu4']], inplace=True)
