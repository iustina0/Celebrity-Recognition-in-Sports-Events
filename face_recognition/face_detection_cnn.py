import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Define the fully connected layers
        self.fc1 = nn.Linear(32 * 56 * 56, 256)  # Adjust input size based on your input image dimensions
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Apply convolutional layers with activation functions and pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Flatten the output of the last convolutional layer
        x = x.view(-1, 32 * 56 * 56)  # Adjust the size based on your input image dimensions
        # Apply fully connected layers with activation functions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage:
# Instantiate the CNN model
model = SimpleCNN(num_classes=10)  # Specify the number of output classes
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
