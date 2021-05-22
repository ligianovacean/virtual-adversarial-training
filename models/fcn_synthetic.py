import torch
import torch.nn.functional as F


class FCNSynthetic(torch.nn.Module):
    def __init__(self):
        super(FCNSynthetic, self).__init__()

        self.layer1 = torch.nn.Linear(100, 100, bias=True)
        self.layer2 = torch.nn.Linear(100, 100, bias=True)
        self.output_layer = torch.nn.Linear(100, 2, bias=True)

    def forward(self, X):
        X = self.layer1(X)
        X = torch.relu(X)
        X = self.layer2(X)
        X = torch.relu(X)
        X = self.output_layer(X)
        #X = F.softmax(X)

        return X

