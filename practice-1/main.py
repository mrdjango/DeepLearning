import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Define the transform to convert images to tensors
transform = transforms.ToTensor()

# Load the MNIST training and test datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Define the number of input nodes
# Each image in MNIST dataset has 28x28 pixels, so the input size is 28*28=784
input_size = 28 * 28

# Define the number of output nodes
# There are 10 classes (digits 0 through 9) in the MNIST dataset
output_size = 10

print("Number of input nodes:", input_size)
print("Number of output nodes:", output_size)
