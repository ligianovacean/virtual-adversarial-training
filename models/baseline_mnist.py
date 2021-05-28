import torch
import torch.nn.functional as F 
from torch import nn


class MNISTBaselineFNN(nn.Module):
    def __init__(self):
        super(MNISTBaselineFNN, self).__init__()

        self.fc1 = nn.Linear(28*28, 400, bias=True)
        self.fc2 = nn.Linear(400, 200, bias=True)
        self.fc3 = nn.Linear(200, 10, bias=True)

        self.bn1 = nn.BatchNorm1d(num_features=400)
        self.bn2 = nn.BatchNorm1d(num_features=200)


    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x