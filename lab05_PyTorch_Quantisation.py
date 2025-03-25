# Install specific versions of PyTorch and torchvision (required for compatibility)
!pip3 install torch==1.5.0 torchvision==1.6.0

# Import core PyTorch and torchvision modules for model building, datasets, and training
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub  # Used for quantizing and dequantizing tensors

# Define a transform pipeline to convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),                   # Converts image to tensor
    transforms.Normalize((0.5,), (0.5,))     # Normalize to [-1, 1] range
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

# Utility class for tracking average values like loss and accuracy during training
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Resets all stats to zero"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates with a new value and sample size n"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Returns formatted string output"""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# Compute Top-1 classification accuracy for a batch
def accuracy(output, target):
    """Computes the top-1 accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)  # Get index of max log-probability
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_one = correct[:1].view(-1).float().sum(0, keepdim=True)
        return correct_one.mul_(100.0 / batch_size).item()

# Utility function to measure model file size (saved as .pt temporarily)
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

# Transfers weights from a regular model to a quantized model structure
def load_model(quantized_model, model):
    state_dict = model.state_dict()
    model = model.to('cpu')
    quantized_model.load_state_dict(state_dict)

# Fuses layers like Conv + ReLU, Linear + ReLU to optimize for quantization
def fuse_modules(model):
    torch.quantization.fuse_modules(model, [['conv1', 'relu1'],
                                            ['conv2', 'relu2'],
                                            ['fc1', 'relu3'],
                                            ['fc2', 'relu4']], inplace=True)
