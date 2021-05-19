import torch
import torch.nn.functional as F 
from torch import nn

class MNISTModelFCN(nn.Module):
    def __init__(self):
        super(MNISTModelFCN, self).__init__()

        self.fc1 = nn.Linear(28*28, 100, bias=True)
        self.fc2 = nn.Linear(100, 50, bias=True)
        self.fc3 = nn.Linear(50, 10, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)

        x = F.log_softmax(x)

        return x
