import torch
import torch.nn.functional as F 
from torch import nn


class MNISTModelFCN(nn.Module):
    def __init__(self):
        super(MNISTModelFCN, self).__init__()

        self.fc1 = nn.Linear(28*28, 500, bias=True)
        self.fc2 = nn.Linear(500, 100, bias=True)
        self.fc3 = nn.Linear(100, 10, bias=True)

        self.bn1 = nn.BatchNorm1d(num_features=500)
        self.bn2 = nn.BatchNorm1d(num_features=100)
        self.bn3 = nn.BatchNorm1d(num_features=10)


    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))

        return x
