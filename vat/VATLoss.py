import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from helpers.log_utils import write_images


PROBABILITY = 0.005


class VATLoss(nn.Module):

    def __init__(self, epsilon=1.0, iterations=2, xi=10.0):
        super().__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.xi = xi

    def get_unit_vector(self, d): 
        # d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        norm = torch.norm(d, dim=1, keepdim=True)
        d /= norm + 1e-8
        return d

    def generate_vap(self, X, network: nn.Module, prediction, device):
        d = torch.rand(X.shape).sub(0.5)
        d = self.get_unit_vector(d)

        # d <- unitvector(H*d)
        for _ in range(self.iterations):
            r = d * self.xi
            r = r.to(device)
            r.requires_grad_()
            perturbation_prediction = network(X + r)
            perturbation_prediction = F.log_softmax(perturbation_prediction, dim=1)
            kl_div = F.kl_div(perturbation_prediction, prediction, reduction='batchmean')
            kl_div.backward()
            gradients = r.grad
            d = self.get_unit_vector(gradients)
            network.zero_grad()

        return self.epsilon * d


    def forward(self, X, network: nn.Module, device, writer):
        with torch.no_grad():
            prediction = network(X)
            prediction = F.softmax(prediction, dim=1)

        r_vadv = self.generate_vap(X, network, prediction, device)

        if random.random() < PROBABILITY:
            write_images(X, r_vadv, writer)

        prediction_rvadv = network(X + r_vadv.detach())
        prediction_rvadv = F.log_softmax(prediction_rvadv, dim=1)
        lds = F.kl_div(prediction_rvadv, prediction, reduction='batchmean')

        return lds
