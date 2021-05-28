import torch
import torch.nn.functional as F 
from torch import nn


class MNISTSupervisedFNN(nn.Module):
    def __init__(self):
        super(MNISTSupervisedFNN, self).__init__()

        self.fc1 = nn.Linear(28*28, 1200, bias=True)
        self.fc2 = nn.Linear(1200, 600, bias=True)
        self.fc3 = nn.Linear(600, 10, bias=True)

        self.bn1 = nn.BatchNorm1d(num_features=1200)
        self.bn2 = nn.BatchNorm1d(num_features=600)


    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x
