# Import standard libraries
import time                       # Used for measuring FPS and logging performance

# Import deep learning and image utilities
import torch                      # PyTorch for model loading and inference
import numpy as np                # Numerical operations
from torchvision import models, transforms  # Torchvision models and image preprocessing
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights  # For quantized MobileNetV2

# Import OpenCV and PIL for image capture and handling
import cv2
from PIL import Image

#%% Configuration

quantize = False  # Set to True if you want to use the quantized version of MobileNetV2

# If quantization is enabled, set the quantized engine to 'qnnpack' (required by PyTorch)
if quantize:
    torch.backends.quantized.engine = 'qnnpack'

#%% Initialize Webcam

cap = cv2.VideoCapture(0)  # Open the default camera (device 0)

# Set the frame dimensions to 224x224 — required by MobileNetV2
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)  # Optional: set a high FPS for smoother real-time experience

#%% Preprocessing Pipeline

# Define preprocessing to match ImageNet-trained MobileNetV2 input:
# 1. Convert image to tensor
# 2. Normalize using ImageNet mean & std
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

#%% Load Model and Class Labels

# Load default weights for quantized MobileNetV2 model
weights = MobileNet_V2_QuantizedWeights.DEFAULT

# Load category labels from model metadata (ImageNet classes)
classes = weights.meta["categories"]

# Load the model with pretrained weights
net = models.quantization.mobilenet_v2(pretrained=True, quantize=quantize)

#%% Performance Tracking

started = time.time()       # Start time for total run
last_logged = time.time()   # Last time FPS was logged
frame_count = 0             # Frames processed since last log

#%% Inference Loop

with torch.no_grad():  # Disable gradient calculation for faster inference
    while True:
        # Read frame from webcam
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")

        # Convert OpenCV's BGR format to RGB (as expected by PIL and torchvision)
        image = image[:, :, [2, 1, 0]]  # Equivalent to cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # (Optional) PIL conversion step, can be used if needed for PIL-based preprocessing
        permuted = image  # Just renaming for clarity

        # Preprocess the frame (normalize + to tensor)
        input_tensor = preprocess(image)

        # Add batch dimension: (C, H, W) → (1, C, H, W)
        input_batch = input_tensor.unsqueeze(0)

        # Run inference on the input batch
        output = net(input_batch)

        # Uncomment the below block to print top-10 predictions with confidence values
        """
        top = list(enumerate(output[0].softmax(dim=0)))
        top.sort(key=lambda x: x[1], reverse=True)
        for idx, val in top[:10]:
            print(f"{val.item()*100:.2f}% {classes[idx]}")
        print(f"========================================================================")
        """

        # Performance logging (frames per second)
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            fps = frame_count / (now - last_logged)
            print(f"============= {fps:.2f} fps =============")
            last_logged = now
            frame_count = 0

'''
Explanation of Key Concepts:

- quantize = False | Disables quantization, enabling it uses a more efficient int8 model.

- MobileNet_V2_QuantizedWeights | Predefined weight set with label metadata for MobileNetV2.

- preprocess | converts image to tensor and normalizes it like the model expects.

- cv2.VideoCapture(0) | Starts capturing from the default camera.

- softmax + argmax | Converts raw scores into probabilities and selects the top class.
'''