import torch
import torch.nn.functional as F


class CNNSynthetic(torch.nn.Module):
    def __init__(self):
        super(CNNSynthetic, self).__init__()

        self.conv1 = torch.nn.Conv1d(1, 2, kernel_size=1, padding=0)
        self.bn1 = torch.nn.BatchNorm1d(2)

        self.conv2 = torch.nn.Conv1d(2, 8, 5, padding=2)
        self.bn2 = torch.nn.BatchNorm1d(8)

        self.fcn1 = torch.nn.Linear(8*25, 8*25)

        self.output_layer = torch.nn.Linear(8*25, 2)

    def forward(self, x):
        batch_size = x.size()[0]
        x = torch.reshape(x, (batch_size, 1, 100))

        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.max_pool1d(x, 2, stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = F.max_pool1d(x, 2, stride=2)

        x = x.view(-1, 8*25)

        x = self.fcn1(x)
        x = F.relu(x)

        x = self.output_layer(x)
        x = F.log_softmax(x)

        return x