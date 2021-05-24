from torch import nn
import torch
import torch.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1) 
        self.pool1 = nn.MaxPool2d(2, stride=2) 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1) 
        self.pool2 = nn.MaxPool2d(2, stride=2) 
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1) 
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    
    def forward(self, x):
        x = x.view(-1, 1, 32, 32)

        x = torch.relu(self.conv1(x)) # 6x28x28
        x = self.pool1(x) # 6x14x14
        x = torch.relu(self.conv2(x)) # 16x10x10
        x = self.pool2(x) # 16x5x5
        x = torch.relu(self.conv3(x)) # 120x1x1
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x)) # 84
        x = self.fc2(x) # 10

        return x


